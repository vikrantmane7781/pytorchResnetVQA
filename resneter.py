import sys,os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
sys.path.append(os.path.realpath('./pytorch-vqa'))
sys.path.append(os.path.realpath('./pytorch-resnet'))
import model
import resnet  # from pytorch-resnet    

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

saved_state = torch.load('./2017-08-04_00.55.19.pth', map_location=device)
tokens = len(saved_state['vocab']['question']) + 1
saved_state.keys()
print("KEYS==",saved_state.keys())

# Load the predefined model
vqa_net = torch.nn.DataParallel(model.Net(tokens))
vqa_net.load_state_dict(saved_state['weights'])
vqa_net.to(device)
vqa_net.eval()

second="static/assets/images/"  
imagePath="static/assets/images/"

files = [ file_path  for _, _, file_path in os.walk(second)]
for file_name in files[0]:
    name=file_name
print("name from op",name)
image_file=imagePath+name

def get_transform(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Resize(int(target_size / central_fraction)),
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

vocab = saved_state['vocab']
vocab.keys()  # dict_keys(['question', 'answer'])
vocabList=list(vocab['question'].items())[:5] 
qtoken_to_index = vocab['question']
QUESTION_LENGTH_MAX = 30 # say...

def encode_question(question_str):
    """ Turn a question into a vector of indices and a question length """
    question_arr = question_str.lower().split(' ')
    #vec = torch.zeros(QUESTION_LENGTH_MAX).long()
    vec = torch.zeros(len(question_arr)).long()  
    for i, token in enumerate(question_arr):
        vec[i] = qtoken_to_index.get(token, 0)
    return vec.to(device), torch.tensor( len(question_arr) ).to(device)

answer_words = ['UNDEF'] * len(vocab['answer'])
for w,idx in vocab['answer'].items():
    answer_words[idx]=w
len(answer_words), answer_words[:10]
quee="how many humans are there?"
q, q_len = encode_question(quee)
v0 = resnet_layer4.image_to_features(image_file)
ans = vqa_net(v0, q.unsqueeze(0), q_len.unsqueeze(0))
ans.data.cpu()[0:30]
_, answer_idx = ans.data.cpu().max(dim=1)
print(answer_idx)
finalAnswer=answer_words[answer_idx]
print("ANSWER===",finalAnswer)
image_features = resnet_layer4.image_to_features(image_file)
print(image_features)