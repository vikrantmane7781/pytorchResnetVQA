import os,sys 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from flask import Flask, render_template, request, Markup
from werkzeug.utils import secure_filename 
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
sys.path.append(os.path.realpath('./pytorch-vqa'))
sys.path.append(os.path.realpath('./pytorch-resnet'))
import model
import resnet  # from pytorch-resnet

#-----------------------------------
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#----------------------------------
imagePath="static/assets/images/"
second="static/assets/images/"

#-------- for model use -------------



## Model load-------------------------------------------------------------
saved_state = torch.load('./2017-08-04_00.55.19.pth', map_location=device)
tokens = len(saved_state['vocab']['question']) + 1


# Load the predefined model
vqa_net = torch.nn.DataParallel(model.Net(tokens))
vqa_net.load_state_dict(saved_state['weights'])
vqa_net.to(device)
vqa_net.eval()
#print("EVALS==",vqa_net.eval())
##-----------------------------------------------------------------------

##  main methods
def get_transform(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Scale(int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


    
class ResNetLayer4(torch.nn.Module):
    def __init__(self):
        super(ResNetLayer4, self).__init__()
        self.model = resnet.resnet152(pretrained=True)
        
        # from  visual_qa_analysis/config.py
        image_size = 448  # scale shorter end of image to this size and centre crop
        #output_size = image_size // 32  # size of the feature maps after processing through a network
        output_features = 2048  # number of feature maps thereof
        central_fraction = 0.875 # only take this much of the centre when scaling and centre cropping

        self.transform = get_transform(image_size, central_fraction)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer
    
    def image_to_features(self, img_file):
        img = Image.open(img_file).convert('RGB')
        img_transformed = self.transform(img)
        img_batch = img_transformed.unsqueeze(0).to(device)
        return self.forward(img_batch) 
    
resnet_layer4 = ResNetLayer4().to(device)

        ##vocab build
vocab = saved_state['vocab']
vocab.keys()  # dict_keys(['question', 'answer'])
vocabList=list(vocab['question'].items())[:5] 
qtoken_to_index = vocab['question']
QUESTION_LENGTH_MAX = 30 # say...

answer_words = ['UNDEF'] * len(vocab['answer'])
for w,idx in vocab['answer'].items():
    answer_words[idx]=w
len(answer_words), answer_words[:10]

def encode_question(question_str):
    """ Turn a question into a vector of indices and a question length """
    question_arr = question_str.lower().split(' ')
    vec = torch.zeros(len(question_arr)).long()  
    for i, token in enumerate(question_arr):
        vec[i] = qtoken_to_index.get(token, 0)
    return vec.to(device), torch.tensor( len(question_arr) ).to(device)




#------------------------ONE FOR ALL -----------------------------
def vqa_single_softmax(im_features, q_str):
    q, q_len = encode_question(q_str)
    ans = vqa_net(im_features, q.unsqueeze(0), q_len.unsqueeze(0))
    return ans.data.cpu()
#--------------------------------------------------------------

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def FirstMain():
    image_name=""
    
    if request.method == 'POST':
        
        for f in os.listdir(second):
            os.remove(os.path.join(second,f))
        f=request.files['file']
        
        path=imagePath+f.filename
        image_name=f.filename
        f.save(path)

        return render_template('info.html',image_name=image_name)
    else:
        name=""
        files = [ file_path  for _, _, file_path in os.walk(second)]
        for file_name in files[0]:
            name=file_name
            #print(name)
        return render_template('info.html',file=name)

### question answer
@app.route('/output', methods=['GET', 'POST'])
def meg():
    name=""
    ans_dict=[]
    files = [ file_path  for _, _, file_path in os.walk(second)]
    for file_name in files[0]:
        name=file_name
    image_file=imagePath+name
    
    #npname=os.path.splitext(name)[0]
    if request.method == 'POST':
        quesName=request.form.get("final")
        sec_ques_name=quesName
        ##---- start model---------
        print(image_file)
        v0 = resnet_layer4.image_to_features(image_file)
        q, q_len = encode_question(quesName)
        ans = vqa_net(v0, q.unsqueeze(0), q_len.unsqueeze(0))
        ans.data.cpu()[0:30]
        _, answer_idx = ans.data.cpu().max(dim=1)
        fnAns=answer_words[answer_idx]
        #image_features = resnet_layer4.image_to_features(image_file)
        _, answer_true = vqa_single_softmax(v0, quesName).max(dim=1)
        print((answer_words[ answer_true ]+' '*8)[:8]+" <- "+quesName)
        
        while True:
            question_arr = quesName.lower().split(' ')
            score_best, q_best = None, ''
            for i, word_omit in enumerate(question_arr):
                question_str = ' '.join( question_arr[:i]+question_arr[i+1:] )
                score, answer_idx = vqa_single_softmax(v0, question_str).max(dim=1)
                if answer_idx==answer_true:
                    dictor=(answer_words[ answer_idx ]+' '*8)[:8]+" <== "+question_str
                    ans_dict.append(dictor)
                    if (score_best is None or score>score_best):
                        score_best, quesName = score, question_str
            
            if score_best is None or len(quesName)==0: break

        print(ans_dict)
        ##------- end model here---------
        return render_template('info.html',quesName=sec_ques_name,file=name,fnAns=fnAns,ans_dict=ans_dict)
    else:
        sec_ques_name=""
        return render_template('info.html',quesName=sec_ques_name,file=name,graph="")

if __name__=="__main__":

    app.run()
