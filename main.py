import src.ImgSim.image_similarity as imgsim

ImgSim = imgsim.Img2Vec('resnet50', weights='DEFAULT')
# observe several of the class attributes post-initialisation
print(ImgSim.architecture)
print(ImgSim.weights)
print(ImgSim.transform)
print(ImgSim.device)
print(ImgSim.model)
print(ImgSim.embed)

# ImgSim.embed_dataset("C://fergu/Documents/PersonalProjects/Image_Clustering/data/all_img/")
ImgSim.embed_dataset("./gen_set")
ImgSim.dataset

target_file = "./new_datasets/23RS030762KF_1_100.jpg.png"
comp_file = "./new_datasets/23RS030762KF_1_100.jpg_generation.png"
import torch.nn as nn
import torch

target_vector = ImgSim.embed_image(target_file)
comp_vector = ImgSim.embed_image(comp_file)

# initiate computation of consine similarity
cosine = nn.CosineSimilarity(dim=1)


dotProduct = torch.dot(torch.flatten(target_vector), torch.flatten(comp_vector))

# iteratively store similarity of stored images to target image
sim = cosine(comp_vector, target_vector)[0].item()
print(sim)
print(dotProduct.item())


dict = ImgSim.similar_images("./new_datasets/23RS030762KF_1_100.jpg.png",n=5)

print(dict)
# 20OE010370KI_1_100.jpg_generation