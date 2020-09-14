import numpy as np 
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d, Axes3D

def vis_detail(kp4_e_array, kp6_e_array, max_dis = 10):
    num_ins = kp6_e_array.flatten().shape[0]
    print(np.where(kp4_e_array < max_dis)[0].shape[0]/num_ins)
    print(np.where(kp6_e_array < max_dis)[0].shape[0]/num_ins)
    #indx = np.where(kp6_e_array > 20)[0]
    fig, axes = plt.subplots(2)
    axes[0].hist(kp4_e_array.flatten(), bins = 10, range = (0,100), weights=np.ones(num_ins)/num_ins)
    axes[0].set_title("Error distributioin for keypoints in C4")
    axes[1].hist(kp6_e_array.flatten(), bins = 10, range = (0,100), weights=np.ones(num_ins)/num_ins)
    axes[1].set_title("Error distributioin for keypoints in C6")
    fig.show()

    indx4 = np.where(kp4_e_array > max_dis)[0]
    kpt4_indx = np.divmod(indx4, 23)[1]

    indx6 = np.where(kp6_e_array > max_dis)[0]
    kpt6_indx = np.divmod(indx6, 23)[1]

    fig, axes = plt.subplots(2, figsize=(5,10))
    axes[0].hist(kpt4_indx,  weights=np.ones(indx4.shape)/num_ins)
    axes[0].set_title("Outlier percentage for keypoints in C4, max distance {}".format(max_dis))
    axes[1].hist(kpt6_indx, weights=np.ones(indx6.shape)/num_ins)
    axes[1].set_title("Outlier percentage for keypoints in C6, max distance {}".format(max_dis))
    fig.show()
    
    
    print("Average reprojection distance in C4: {}".format(np.average(kp4_e_array)))
    print("Average reprojection distance in C6: {}".format(np.average(kp6_e_array)))

    print("Minimum reprojection distance in C4: {}".format(np.min(kp4_e_array)))
    print("Minimum reprojection distance in C6: {}".format(np.min(kp6_e_array)))

    print("Maximum reprojection distance in C4: {}".format(np.max(kp4_e_array)))
    print("Maximum reprojection distance in C6: {}".format(np.max(kp6_e_array)))

    print("Variance of reprojection distance in C4: {}".format(np.sqrt(np.var(kp4_e_array.flatten()))))
    print("Variance of reprojection distance in C4: {}".format(np.sqrt(np.var(kp6_e_array.flatten()))))

    input("Any key....")
    print("OK")

def VisualizeOne3D(pts3D_person,co,ax, revert = True):
    """
        this function visualizes one person in colorful 3D skeleton, with different colors
        represent different joints, the firsy input is 3 by 23 numpy array
        the second input is the color of skeleton
        input follows COCO convention which is
        {0,  "Nose"}, 
        {1,  "LEye"}, 
        {2,  "REye"}, 
        {3,  "LEar"}, 
        {4,  "REar"}, 
        {5,  "LShoulder"}, 
        {6,  "RShoulder"}, 
        {7,  "LElbow"}, 
        {8,  "RElbow"}, 
        {9,  "LWrist"},
        {10, "RWrist"}, 
        {11, "LHip"}, 
        {12, "RHip"}, 
        {13, "LKnee"}, 
        {14, "Rknee"},
        {15, "LAnkle"}, 
        {16, "RAnkle"}, 
        {17, "LBigToe"}, 
        {18, "LSmallToe"}, 
        {19, "LHeel"}, 
        {20, "RBigToe"}, 
        {21, "RSmallToe"},
        {22, "RHeel"}, 
    """ 
    pts3D = pts3D_person
    if revert:
        pts3D = np.array([pts3D[0,:],pts3D[2,:],-pts3D[1,:]])
    colormap = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,140/255,0),(0,100/255,0)]
    #colormap = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,140/255,0],[0,100/255,0]])
   
    #find neck
    neck3D_x = (pts3D[0, 5] + pts3D[0, 6])*0.5
    neck3D_y = (pts3D[1, 5] + pts3D[1, 6])*0.5
    neck3D_z = (pts3D[2, 5] + pts3D[2, 6])*0.5
    ax.scatter(neck3D_x, neck3D_y, neck3D_z, c=colormap[6], marker='o')#Neck
    #find head
    p1 = ((pts3D[0, 1] + pts3D[0, 2])*0.5,(pts3D[1, 1] + pts3D[1, 2])*0.5,(pts3D[2, 1] + pts3D[2, 2])*0.5)
    p2 = ((pts3D[0, 3] + pts3D[0, 4])*0.5,(pts3D[1, 3] + pts3D[1, 4])*0.5,(pts3D[2, 3] + pts3D[2, 4])*0.5)
    p3 = (pts3D[0, 0], pts3D[1, 0],pts3D[2, 0])
    p4 = (0,0,0)
    p_h = SingularityElimination(p1,p2,p3,p4)
    ax.scatter(p_h[0], p_h[1], p_h[2], c=colormap[0], marker='o')#head
    
    ax.scatter(pts3D[0, 5], pts3D[1, 5], pts3D[2, 5], c=colormap[7], marker='o') #LShoulder
    ax.scatter(pts3D[0, 6], pts3D[1, 6], pts3D[2, 6], c=colormap[7], marker='o') #Rshoulder
    ax.scatter(pts3D[0, 7], pts3D[1, 7], pts3D[2, 7], c=colormap[2], marker='o') #LElbow
    ax.scatter(pts3D[0, 8], pts3D[1, 8], pts3D[2, 8], c=colormap[2], marker='o') #RElbow
    ax.scatter(pts3D[0, 9], pts3D[1, 9], pts3D[2, 9], c=colormap[3], marker='o') #LWrist
    ax.scatter(pts3D[0, 10], pts3D[1, 10], pts3D[2, 10], c=colormap[3], marker='o') #RWrist
    ax.scatter(pts3D[0, 11], pts3D[1, 11], pts3D[2, 11], c=colormap[4], marker='o') #LHip
    ax.scatter(pts3D[0, 12], pts3D[1, 12], pts3D[2, 12], c=colormap[4], marker='o') #RHip
    ax.scatter(pts3D[0, 13], pts3D[1, 13], pts3D[2, 13], c=colormap[5], marker='o') #Lknee
    ax.scatter(pts3D[0, 14], pts3D[1, 14], pts3D[2, 14], c=colormap[5], marker='o') #Rknee
     
    #feet
    p_f_L = SingularityElimination(pts3D[:,[15]],pts3D[:,[17]],pts3D[:,[18]],pts3D[:,[19]])
    p_f_R = SingularityElimination(pts3D[:,[16]],pts3D[:,[20]],pts3D[:,[21]],pts3D[:,[22]])
    ax.scatter(p_f_L[0], p_f_L[1], p_f_L[2], c=colormap[1], marker='o')
    ax.scatter(p_f_R[0], p_f_R[1], p_f_R[2], c=colormap[1], marker='o') 
    
    #connect skeleton 
    ConnectSkeleton(11,13,co,pts3D,ax)  
    ConnectSkeleton(12,14,co,pts3D,ax) 
    ConnectSkeleton(5,6,co,pts3D,ax)
    ConnectSkeleton(5,7,co,pts3D,ax)  
    ConnectSkeleton(7,9,co,pts3D,ax) 
    ConnectSkeleton(6,8,co,pts3D,ax)
    ConnectSkeleton(8,10,co,pts3D,ax)


    x = [p_h[0], neck3D_x]
    y = [p_h[1], neck3D_y]
    z = [p_h[2], neck3D_z]
    ax.plot(x,y,z,color=co )
    x = [pts3D[0, 11], neck3D_x]
    y = [pts3D[1, 11], neck3D_y]
    z = [pts3D[2, 11], neck3D_z]
    ax.plot(x,y,z,color=co )
    x = [pts3D[0, 12], neck3D_x]
    y = [pts3D[1, 12], neck3D_y]
    z = [pts3D[2, 12], neck3D_z]
    ax.plot(x,y,z,color=co )
    x = [pts3D[0, 13], p_f_L[0]]
    y = [pts3D[1, 13], p_f_L[1]]
    z = [pts3D[2, 13], p_f_L[2]]
    ax.plot(x,y,z,color=co )
    x = [pts3D[0, 14], p_f_R[0]]
    y = [pts3D[1, 14], p_f_R[1]]
    z = [pts3D[2, 14], p_f_R[2]]
    ax.plot(x,y,z,color=co )

def SingularityElimination(p1,p2,p3,p4):
    d12 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2
    d23 = (p2[0] - p3[0])**2 + (p2[1] - p3[1])**2 + (p2[2] - p3[2])**2
    d13 = (p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2
    d14 = (p1[0] - p4[0])**2 + (p1[1] - p4[1])**2 + (p1[2] - p4[2])**2
    d24 = (p2[0] - p4[0])**2 + (p2[1] - p4[1])**2 + (p2[2] - p4[2])**2
    d34 = (p3[0] - p4[0])**2 + (p3[1] - p4[1])**2 + (p3[2] - p4[2])**2
    d = np.array([d12,d13,d14,d23,d24,d34])
    min_ind_s = np.unravel_index(np.argmin(d, axis=None), d.shape)
    min_ind = min_ind_s[0]    
    if min_ind == 0:
        p_f = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2]))
    if min_ind == 1:
        p_f = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]), 0.5 * (p1[2] + p3[2]))
    if min_ind == 2:
        p_f = (0.5 * (p1[0] + p4[0]), 0.5 * (p1[1] + p4[1]), 0.5 * (p1[2] + p4[2]))
    if min_ind == 3:
        p_f = (0.5 * (p2[0] + p3[0]), 0.5 * (p2[1] + p3[1]), 0.5 * (p2[2] + p3[2]))
    if min_ind == 4:
        p_f = (0.5 * (p2[0] + p4[0]), 0.5 * (p2[1] + p4[1]), 0.5 * (p2[2] + p4[2]))
    if min_ind == 5:
        p_f = (0.5 * (p3[0] + p4[0]), 0.5 * (p4[1] + p4[1]), 0.5 * (p3[2] + p4[2]))
    return p_f

def ConnectSkeleton(i,j,co,pts3D,ax):
    x = [pts3D[0, i], pts3D[0, j]]
    y = [pts3D[1, i], pts3D[1, j]]
    z  = [pts3D[2, i], pts3D[2, j]]
    ax.plot(x,y,z,color=co )
    return x,y,z
       
def VisualizeAll3D(pts3D,ax):
    """
        this function visualizes all 3D skeletons from one image
        the form if pts3D is 4 by n numpy array, rows 0,1,2 are x,y,z, row3 is always 1
    """
    num_people = pts3D.shape[1] / 23
    for counter in range (0,int(num_people)):
        start = counter * 23
        end = start + 23
        pts3D_person = pts3D[0:3,start:end]
        VisualizeOne3D(pts3D_person,(1,0,0),ax)