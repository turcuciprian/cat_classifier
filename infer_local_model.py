from fastai.learner import load_learner
from PIL import Image
from torchvision import transforms

def is_cat(x): return x[0].isupper()

learn = load_learner("/home/ciprian/Documents/work/cat_vs_dog_fastai_clasifier/catVsDog.pkl")
img = Image.open('test.jpg')

# Define the transformations: resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Re  x xsize to the size your model expects, e.g. 224x224 for models like ResNet
    transforms.ToTensor()])

img_t = transform(img)

is_cat,_,probs = learn.predict(img)

print(f"Is this a cat?: {is_cat}.")