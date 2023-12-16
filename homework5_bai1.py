import numpy as np
import matplotlib.pyplot as plt

def read_data(data_path, size):
    with open(data_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint8, count=size*size)
        return np.reshape(data, (size, size))

def stretch(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0

data_path = 'dataset/salesman.sec'
size = 256
read = read_data(data_path, size)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(read, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Original Image', fontsize=18)
X = np.zeros((262, 262))
X[4:260, 4:260] = read
Y = np.zeros((262, 262))
for row in range(4, 261):
    for col in range(4, 261):
        Y[row, col] = np.sum(X[row-3:row+4, col-3:col+4]) / 49
Y2 = stretch(Y[4:260, 4:260])
plt.subplot(1,2, 2)
plt.imshow(Y2, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Filtered Image', fontsize=18)
plt.show()

#Cau b
plt.figure(figsize=(12, 6))
plt.subplot(2, 4, 1)
plt.imshow(read, cmap='gray')
plt.title('Original image', fontsize=12)
plt.axis('image')
plt.axis('off')
size1 = 256 + 128 - 1
pX = np.zeros((size1, size1))
pX[:256, :256] = read
plt.subplot(2, 4, 2)
plt.imshow(pX, cmap='gray')
plt.title('Zero Padded', fontsize=12)
plt.axis('image')
plt.axis('off')
H = np.zeros((128, 128))
H[62:69, 62:69] = 1 / 49
pH = np.zeros((size1, size1))
pH[:128, :128] = H
plt.subplot(2, 4, 3)
plt.imshow(pH, cmap='gray')
plt.title('Zero Padded Impulse Resp', fontsize=12)
plt.axis('image')
plt.axis('off')
pXtilde = np.fft.fft2(pX)
pHtilde = np.fft.fft2(pH)
pXtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(pXtilde)))
plt.subplot(2, 4, 4)
plt.imshow(pXtildeDisplay, cmap='gray')
plt.title('Log-mag spectrum zero pad', fontsize=12)
plt.axis('image')
plt.axis('off')
pHtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(pHtilde)))
plt.subplot(2, 4, 5)
plt.imshow(pHtildeDisplay, cmap='gray')
plt.title('Log-magnitude spectrum H', fontsize=12)
plt.axis('image')
plt.axis('off')
pYtilde = pXtilde * pHtilde
pY = np.fft.ifft2(pYtilde)
pYtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(pYtilde)))
plt.subplot(2, 4, 6)
plt.imshow(pYtildeDisplay, cmap='gray')
plt.title('Log-magnitude spectrum of result', fontsize=12)
plt.axis('image')
plt.axis('off')
plt.subplot(2, 4, 7)
plt.imshow(np.real(pY), cmap='gray')
plt.title('Zero Padded Result', fontsize=12)
plt.axis('image')
plt.axis('off')
Y = np.real(pY[64:320, 64:320])
plt.subplot(2, 4, 8)
plt.imshow(Y, cmap='gray')
plt.title('Final Filtered Image', fontsize=12)
plt.axis('image')
plt.axis('off')
plt.show()

#Cau c
def stretch(x):
    xMax = np.max(x)
    xMin = np.min(x)
    scale = 255.0 / (xMax - xMin)
    y = np.round((x - xMin) * scale)
    return y.astype(np.uint8)
H1 = np.zeros((256, 256))
H1[126:133, 126:133] = 1/49
H2 = np.fft.fftshift(H1)
print(H2.shape)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(stretch(read), cmap='gray')
plt.title('Zero Phase Impulse Resp', fontsize=18)
plt.axis('image')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(stretch(H2), cmap='gray')
plt.title('Zero Phase Impulse Resp', fontsize=18)
plt.axis('image')
plt.axis('off')
pX = np.zeros((512, 512))
pX[:256, :256] = read
pH2 = np.zeros((512, 512))
pH2[:128, :128] = H2[:128, :128]
pH2[:128, 385:512] = H2[:128, 129:256]
pH2[385:512, :128] = H2[129:256, :128]
pH2[385:512, 385:512] = H2[129:256, 129:256]
plt.subplot(2, 2, 3)
plt.imshow(stretch(pH2), cmap='gray')
plt.title('Zero Padded zero-phase H', fontsize=18)
plt.axis('image')
plt.axis('off')
Y = np.fft.ifft2(np.fft.fft2(pX) * np.fft.fft2(pH2))
Y = stretch(Y[:256, :256])
plt.subplot(2, 2, 4)
plt.imshow(Y, cmap='gray')
plt.title('Final Filtered Image', fontsize=18)
plt.axis('image')
plt.axis('off')
plt.show()


