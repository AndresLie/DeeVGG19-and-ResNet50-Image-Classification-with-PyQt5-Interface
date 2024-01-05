import cv2,numpy as np

class Closing():
    def __init__(self,image_path):
        self.image_path=image_path
        image=cv2.imread(self.image_path)
        gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        threshold = 127
        binary_image = (gray_image > threshold).astype(np.uint8) * 255

        # Pad the image with zeros based on the kernel size (K=3)
        kernel_size = 3
        padded_image = np.pad(binary_image, pad_width=kernel_size // 2, mode='constant', constant_values=0)

        # Define a 3x3 all-ones structuring element
        structuring_element = np.ones((3, 3), dtype=np.uint8)

        # Perform dilation operation
        eroded_image = np.zeros_like(padded_image)
        for i in range(1, padded_image.shape[0] - 1):
            for j in range(1, padded_image.shape[1] - 1):
                eroded_image[i, j] = np.min(padded_image[i-1:i+2, j-1:j+2] * structuring_element)

        # Perform dilation operation
        dilated_image = np.zeros_like(padded_image)
        for i in range(1, padded_image.shape[0] - 1):
            for j in range(1, padded_image.shape[1] - 1):
                dilated_image[i, j] = np.max(padded_image[i-1:i+2, j-1:j+2] * structuring_element)

        # Perform opening operation (erosion followed by dilation)
        opened_image = dilated_image
        cv2.imshow("Closing",opened_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Opening():
    def __init__(self,image_path):
        self.image_path=image_path
        image=cv2.imread(self.image_path)
        gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        threshold = 127
        binary_image = (gray_image > threshold).astype(np.uint8) * 255

        # Pad the image with zeros based on the kernel size (K=3)
        kernel_size = 3
        padded_image = np.pad(binary_image, pad_width=kernel_size // 2, mode='constant', constant_values=0)

        # Define a 3x3 all-ones structuring element
        structuring_element = np.ones((3, 3), dtype=np.uint8)

        # Perform erosion operation
        eroded_image = np.zeros_like(padded_image)
        for i in range(1, padded_image.shape[0] - 1):
            for j in range(1, padded_image.shape[1] - 1):
                eroded_image[i, j] = np.min(padded_image[i-1:i+2, j-1:j+2] * structuring_element)

        # Perform dilation operation
        dilated_image = np.zeros_like(padded_image)
        for i in range(1, padded_image.shape[0] - 1):
            for j in range(1, padded_image.shape[1] - 1):
                dilated_image[i, j] = np.max(eroded_image[i-1:i+2, j-1:j+2] * structuring_element)


        cv2.imshow("Opening",dilated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()