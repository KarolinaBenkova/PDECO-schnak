import matplotlib.image as mpimg
import numpy as np
from PIL import Image
# Contains functions used for importing experimental data

def get_data(data, bc, nodes, boundary_nodes, gamma):
    """Loads the target data: hat{u} if input is "u", or hat{v} if input is "v".
    Assume square domain so that the image is square.The size of the loaded image 
    should be pixel_dim x pixel_dim, where pixel_dim is the square root of the 
    number of nodes.
    
    Parameters:
        data (str): The type of data to load (the variable, 'u' or 'v').
        bc (str): The boundary condition type ('Neumann').
        nodes (int): The number of nodes in the grid.
        boundary_nodes (list): The list of boundary nodes.
        gamma (int): The gamma value for the data.
    
    Returns:
        numpy.ndarray: A numpy array of shape (pixel_dim, pixel_dim).
    """
    img_path = 'data/'
    pixel_dim = int(np.sqrt(nodes))
    if pixel_dim**2 != nodes:
        raise ValueError(f"{nodes} is not a perfect square.")
    
    if data=='u': # loads \hat{u}
        if bc=='Neumann':
                bc_type= 'NBC'
                if gamma==1000:
                    c = 0.5858
                    d = 1.6204
                elif gamma==100:
                    c = 0.654
                    d = 1.7596
                elif gamma==80:
                    c =  0.5668
                    d = 1.17011
                elif gamma==50:
                    c = 0.5663
                    d = 1.5907
                elif gamma==10:
                    c = 0.4015
                    d = 5.5049
                else:
                    print('Wrong value of gamma')
    elif data=='v': # loads \hat{v}
        if bc=='Neumann':
                bc_type= 'NBC'
                if gamma==1000:
                    c = 0.6498
                    d = 1.0634
                elif gamma==100:
                    c = 0.654
                    d = 0.7596
                elif gamma==80:
                    c = 0.7823
                    d = 1.0982
                elif gamma==50:
                    c = 0.6900
                    d = 1.0542
                elif gamma==10:
                    c = 0.2018
                    d = 0.6302
                else:
                    print('Wrong value of gamma')
    else:
        return print("Error: Input is incorrect.")        
               
    img_name = 'schnak_' + data + '_' + str(gamma) + '_' + bc_type + str(pixel_dim) + '.png'
    img_rgb = mpimg.imread(img_path + img_name)
    
    # Create greyscale image by averaging over RGB (-> values in [0,1])
    img_grey = np.mean(img_rgb, axis=2)
    
    # Pixel values min, max:
    a = np.amin(img_grey)
    b = np.amax(img_grey)

    # Linear transform from [a,b] to [c,d]
    img_t = (d-c)/(b-a) * (img_grey-a) + c
    
    return img_t

def generate_image(data, bc, nodes, gamma):
    """Generates a resized image for the target data: hat{u} if input is "u", or hat{v} if input is "v".
    The generated image will match the mesh size specified by the number of nodes, assuming the domain is square.

    Parameters:
        data (str): The type of data to load (the variable, 'u' or 'v').
        bc (str): The boundary condition type ('Neumann').
        nodes (int): The number of nodes in the grid.
        gamma (int): The gamma value for the data.

    Returns:
        None
    """
    
    pixel_dim = int(np.sqrt(nodes))
    if pixel_dim**2 != nodes:
        raise ValueError(f"{nodes} is not a perfect square.")
        
    if bc=='Neumann': # loads \hat{u}
        bc_type = 'NBC'
    else:
        return print("Error: Input is incorrect.")
    
    img_path = 'data/'
    img_orig = 'schnak_' + data + '_' + str(gamma) + '_' + bc_type + '.png'
    img_new = 'schnak_' + data + '_' + str(gamma) + '_' + bc_type + str(pixel_dim) + '.png'

    # Resize image automatically instead of suplying images of different lengths
    img = Image.open(img_path+img_orig)
    img_re = img.resize((pixel_dim,pixel_dim))
    img_re.save(img_path+img_new)
    print(f'Created image of dimension {pixel_dim} for variable {data}, {gamma=}, and {bc} boundary conditions')    

def uhat(t,T,nodes,boundary_nodes,bc,gamma):
    """
    Interpolation of the target state uhat.
    """
    out = t/T * get_data("u",bc,nodes,boundary_nodes,gamma).reshape(nodes)
    return out

def vhat(t,T,nodes,boundary_nodes,bc,gamma):
    """
    Interpolation of the target state vhat.
    """
    out = t/T * get_data("v",bc,nodes,boundary_nodes,gamma).reshape(nodes)
    return out