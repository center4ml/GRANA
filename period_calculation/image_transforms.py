import numpy as np
import torch
import cv2


def np_batched_radon(image_batch):
    #image_batch:    torch tensor, batch_size x 1 x img_size x img_size
    # squeeze order #1 and transform to numpy

    image_batch = image_batch.squeeze(1).cpu().numpy()

    batch_size, img_size = image_batch.shape[:2]
    if batch_size > 512: # limit batch size to 512 because cv2.warpAffine fails for batch> 512
        return np.concatenate([np_batched_radon(image_batch[i:i+512]) for i in range(0,batch_size,512)], axis=0)
    theta = np.arange(180)
    radon_image = np.zeros((image_batch.shape[0], img_size, len(theta)),
                           dtype='float32')

    for i, angle in enumerate(theta):
        M = cv2.getRotationMatrix2D(((img_size-1)/2.0,(img_size-1)/2.0),angle,1)
        rotated = cv2.warpAffine(np.transpose(image_batch, (1, 2, 0)),M,(img_size,img_size))

        #plt.imshow(rotated[:,:,0])
        #plt.show()

        if batch_size == 1: # cv2.warpAffine cancels batch dimension if equal to 1
          rotated = rotated[:,:, np.newaxis]
        rotated = np.transpose(rotated, (2, 0, 1)) / 224.0
        #rotated = rotated / np.array(255, dtype='float32')
        radon_image[:, :, i] = rotated.sum(axis=1)

    #plot the image

  #  plt.imshow(radon_image[0])
   # plt.show()

    return radon_image


def torch_batched_radon(image_batch, neutral_value):
    #image_batch:                                batch_size x 1 x img_size x img_size
    #np_batched_radon(image_batch - neutral_value)

    image_batch = image_batch - neutral_value   # so the 0 value is neutral


    batch_size = image_batch.shape[0]
    img_size = image_batch.shape[2]

    theta = np.arange(180)   # we don't need torch here, we will evaluate individual angles below

    radon_image = torch.zeros((batch_size, 1, img_size, len(theta)), dtype=torch.float, device=image_batch.device)


    for i, angle in enumerate(theta):
        #M = cv2.getRotationMatrix2D(((img_size-1)/2.0,(img_size-1)/2.0),angle,1)
        #calculate the same rotation matrix but with torch:
        M = torch.tensor(cv2.getRotationMatrix2D(((img_size-1)/2.0,(img_size-1)/2.0),angle,1)).to(image_batch.device, dtype=torch.float32)
        angle = torch.tensor((angle+90)/180.0*np.pi)
        M1 = torch.tensor([[torch.sin(angle), torch.cos(angle), 0],
                    [torch.cos(angle), -torch.sin(angle), 0]]).to(image_batch.device, dtype=torch.float32)


        # we need to add a batch dimension to the rotation matrix
        M1 = M1.repeat(batch_size, 1, 1)

        grid = torch.nn.functional.affine_grid(M1, image_batch.shape, align_corners=False)
        rotated = torch.nn.functional.grid_sample(image_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        rotated = rotated.squeeze(1)

        #plt.imshow(rotated[0].cpu().numpy())
        #plt.show()

        radon_image[:, 0, :, i] = rotated.sum(axis=1) / 224.0 + neutral_value

    #plt.imshow(radon_image[0, 0].cpu().numpy())
    #plt.show()

    return radon_image
