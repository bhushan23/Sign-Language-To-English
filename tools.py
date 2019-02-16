import torch
# def show(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0))) #, interpolation='nearest')
#     
# def show2(img):
#     plt.imshow(img)
#     plt.show()
#     
# def save_image(pic):
#     grid = torchvision.utils.make_grid(pic, nrow=8, padding=2)
#     ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
#     plt.imshow(ndarr)
#     plt.show()
#     #im = Image.fromarray(ndarr)
#     #im.save(path)
# 
# def denorm(x):
#     out = (x + 1) / 2
#     return out.clamp(0, 1)
# 

def test(model, name, data_loader, print_acc = True):

  correct_cnt = 0
  elements    = 0.0
  
  for _, sample in enumerate(data_loader):
    data = sample[0] #.cuda()
    label = torch.tensor(sample[1]) #.cuda()
    
    data = data.cuda()
    label = label.cuda()
    
    pred = model.forward(data)
    # print(pred.shape)
    pred = pred.max(1)[1]
    # print(pred.shape, label.shape)
    cnt  = (pred == label)
    # print(cnt.shape)
    correct_cnt += torch.sum(cnt)
    elements    += data.shape[0]
  acc = (float(correct_cnt.item()) / (float(elements))) * 100.0
  
  if print_acc:
    print(name, ': ', acc)
  return acc
