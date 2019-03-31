import cv2
import numpy as np
import imgaug.augmenters as iaa


class Generator(object):

    def __init__(self):


        self.sequence = iaa.OneOf([iaa.Fliplr(),
                                   iaa.Affine(rotate=20),
                                   iaa.AdditiveGaussianNoise(scale=0.2*255),
                                   iaa.Sharpen(alpha=0.5),
                                   iaa.Multiply((1.2, 1.5))])


    def __read_image_paths_labels(self, annotations_csv_path): 
        """
        The method reads the annotation.csv file and extracts the
        image path and their corresponding labels

        Args:
        - annotations_csv_path (str): a csv file with image paths and labels

        Returns:
        - image_paths 	(list): a list of image paths
        - labes 		(list): a list of labels corresponding to the images
        """

        # intialize some list objects

        
        temp_df = pd.read_csv(annotations_csv_path, index_col=[0])
        labels = temp_df.drop('Path', axis=1).values
        image_paths = temp_df['Path'].tolist()

        return image_paths, labels


    def __read_image(self, path):
        """
        This method reads the images and ensures all returned images
        are 3 channle RGB images

        Args:
        - path 		(str)		: string indicating image path

        Returns:
        - image 	(nd.array)	: 3-d RGB tensor containing image information
        """
        path = 'data/' + path
        image = cv2.imread(path)

        # Convert greyscale image to BGR
        if image.shape[-1] == 1:
            image = np.dstack([image, image, image])

        # Convert BGR image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image



    def __preprocesses_image(self, image, size):
        """
        This method performs required preprocessing on the images
        (resize, clahe, contrast)

        Args:
        - image 	(nd.array)	: string indicating image path
        - size 		(int)		: size to which input array ought to be resized (size, size) 	 

        Returns:
        - image 	(nd.array)	: 3-d RGB tensor containing image information
        """

        # Resize image
        image = cv2.resize(image, (size, size))

        # Normalize image (min-max)
        image = np.float32((image -  np.min(image)) / (np.max(image) - np.min(image) + 0.00001))

        return image



    def generator(self, annotations_csv_path, num_classes, augmentation=False, batch_size=4, size=224):
        """
        This method creates and returns a generator object

        Args:
        - annotations_csv_path 	(str)  		: path towards the annotation file
        - augmentation 			(bool) 		: boolean indicating if augmentation ought to be done
        - batch_size 			(int)		: integer indicating training batch size
        - size 					(int)		: size input images ought to be resized to (size, size)

        Yeilds:
        - image_batch			(nd.aarray) : 4-D tensor of shape (batch_size, size, size, 3)
        - label_batch 			(nd.array) 	:
        """

        # Get image paths and labels
        image_paths, labels = self.__read_image_paths_labels(annotations_csv_path)
        steps = len(image_paths)//batch_size
        
        step = 0
        itr = 0
        
        while True:
        #for itr in range(0, len(image_paths), batch_size):

            # Storing batches of paths and labels in lists
            temp_path = [image_paths[i] for i in range(itr, itr + batch_size)]
            temp_label = [labels[i] for i in range(itr, itr + batch_size)]

            # Create empty tensors for images and labels
            image_batch = np.zeros((batch_size, size, size, 3), dtype=np.float32)
            label_batch = np.zeros((batch_size, num_classes), dtype=np.float32)

            # Keep track of batch size
            count = 0

            for n, path in enumerate(temp_path):

                temp_org_image = self.__read_image(path)
                temp_image = self.__preprocesses_image(temp_org_image, size)

                image_batch[count] = temp_image
                label_batch[count] = temp_label[n]

                # At least two more empty arrays must be available in the 
                # image_batch tensor
                if not temp_label[n][-1] and count < batch_size-2 and augmentation: 

                    aug_image_1 = self.sequence.augment_image(temp_org_image)
                    aug_image_2 = self.sequence.augment_image(temp_org_image)

                    aug_image_1 = self.__preprocesses_image(aug_image_1, size)
                    aug_image_2 = self.__preprocesses_image(aug_image_2, size)

                    image_batch[count+1] = aug_image_1
                    label_batch[count+1] = temp_label[n]

                    image_batch[count+2] = aug_image_2
                    label_batch[count+2] = temp_label[n]

                    count += 3

                else: 
                    count += 1


                if count == batch_size:
                    break
                    
            step += 1
            itr += batch_size

            yield image_batch, label_batch
            
            if step >= steps:
                step = 0
                itr = 0


