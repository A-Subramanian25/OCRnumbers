data_directory=r'C:\Users\AKIL\Downloads\OCR project\mnist'
train_data_path = data_directory + r'\train-images-idx3-ubyte\train-images-idx3-ubyte'
train_data_labels=data_directory + r'\train-labels-idx1-ubyte\train-labels-idx1-ubyte'
test_data_path = data_directory + r'\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte'
test_data_labels=data_directory + r'\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte'

#convert data from byte format to integer format
def byte_to_int(byte_data):
    return int.from_bytes(byte_data,'big')

#############################################################
#reading an image and converting it into an numpy array for processing
debug=True
if debug:
    import numpy as np
    from PIL import Image
    def read_image_data(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)
#############################################################
#reading the file
# 4 bytes of magic number
# 4 bytes of number of images
# 4 bytes of rows
# 4 bytes of columns
# Remaining data about the pixels
def read_image(file_name,no_max_Images=None):
    images=[]
    with open(file_name,'rb') as f:
        _=f.read(4)
        n_images=byte_to_int(f.read(4))
        n_rows=byte_to_int(f.read(4))
        n_columns=byte_to_int(f.read(4))
        if (no_max_Images!=None):
            n_images=no_max_Images
        for i in range(n_images):
            image=[]
            for row in range(n_rows):
                row_l=[]
                for col in range(n_columns):
                    row_l.append(f.read(1))
                image.append(row_l)
            images.append(image)
    return images
    
#############################################################

# reading the labels
# 4 bytes of magic number
# 4 bytes of number of labels
# rest all labels
def read_labels(file_name,no_max_labels=None):
    labels=[]
    with open(file_name,'rb') as f:
        _=f.read(4)
        n_labels=byte_to_int(f.read(4))
        if (no_max_labels!=None):
            n_labels=no_max_labels
        for i in range(n_labels):
            labels.append(f.read(1))
    return labels

#################################################################

#decompressing the data
#flattening the 2d array into a 1d array
def flattening_image(X):
    flatten_list=[]
    for sublist in X:
        for pixel in sublist:
            flatten_list.append(pixel)
    return flatten_list

#################################################################

#extracting the data from the list
#converting the 3d array to a 2d array
def extract_data(X):
    extracted_data=[]
    for image in X:
        extracted_data.append(flattening_image(image))
    return extracted_data

#################################################################

#lazy learners algorithm
#K- Nearest Neighbours algorithm
#k is preferably odd to break the ties in case of equal votes
def KNN(X_train,y_train,X_test,k):
    y_pred=[]
    for test_image in X_test:
        distance=get_distance_for_training_sample(X_train,test_image)
        #lambda notation study about this for writing code in reduced LOC
        sorted_pair_indices=[ 
        pair[0] for pair in sorted (
        enumerate (distance),
        key=lambda x:x[1]
        )]
        candidates=[
            byte_to_int(y_train[idx]) for idx in sorted_pair_indices[:k]
        ]
        top_candidate=get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    return y_pred
#################################################################
#finding the most frequent element
def get_most_frequent_element(candidates):
    return max(candidates,key=candidates.count)
#################################################################
#helper to simplify the process of finding the distance
def get_distance_for_training_sample(X_train,test_img):
    dist=[]
    for image in X_train:
        dist.append(distance_finder(image,test_img))
    return dist

#################################################################
#distance finder
#using euclidean distance
def distance_finder(image,test_image):
    sum=0
    for i in range(0,len(image)):
        sum+=(byte_to_int(image[i])-byte_to_int(test_image[i]))**2
    sum=sum**(0.5)
    return sum
#################################################################
#main function
def main():
    #reading data
    X_train=read_image(train_data_path,60000)
    y_train=read_labels(train_data_labels,60000)
    X_test=read_image(test_data_path,10)
    #X_test=[read_image_data(r'------------------------------------------------')]
    y_test=read_labels(test_data_labels,10)

    #data transformation
    X_train=extract_data(X_train)
    #X_test=extract_data(X_test)
    X_test=extract_data(X_test)
    #print(X_test)
    y_pred=KNN(X_train,y_train,X_test,5)
    correct_predictions=[]
    for i in range(0,len(y_pred)):
        if (y_pred[i]==byte_to_int(y_test[i])):
            correct_predictions.append(1)
    accuracy=sum(correct_predictions)/len(y_test)
    print(accuracy)
    print(y_pred)

if __name__ == '__main__':
    main()


