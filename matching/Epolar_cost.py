############################ cost functin of epolar line ######################################
def epolar_cost_function(list_kp_4,list_kp_6,P4,P6, filtered_index_C4, filtered_index_C6):
    """calculate cost matrix epolar, consider n people altogether in one frame
    Call one function, namely, getEpolar, getTotaldistance
    
    Args:
    -list_kp_4 69*m key points from one image, format: x1, y1, score1, x2, y2, score2, ...
    -list_kp_6 69*n key points from one image, format: x1, y1, score1, x2, y2, score2, ...
    
    Returns:
    -C_epolar a n by m matrix containing cost functions
    -M_index_epolar a n by m matrix containing indexes of matching
    """
    confi_thresh = 0.2 #when the confidence of a particular joint is lower than this threshold, Alpha pose didn't find this joint
    people_num4 = int(len(list_kp_4))
    people_num6 = int(len(list_kp_6))
    C_epolar = np.ones((people_num6, people_num4)) * 10
    M_index_epolar = np.zeros((people_num6, people_num4),dtype = object)
    for i in range(0,people_num6):
        for j in range(0,people_num4):
            M_index_epolar[i,j] = str(filtered_index_C6[i]) + ',' + str(filtered_index_C4[j])
            total_distance = getTotaldistance(list_kp_4[j], list_kp_6[i], P4,P6)
            C_epolar[i,j] = total_distance  
    C_epolar = C_epolar / np.max(C_epolar)
    return C_epolar,M_index_epolar

def getTotaldistance(list_sp4,list_sp6,P4,P6):
    """calculate cost matrix epolar, consider n people altogether in one frame
    Call get_distance to calculate the distance between epolar line and the point with the same name
    
    Args:
    -list_sp4 69 numbers from one image, format: x1, y1, score1, x2, y2, score2, ...
    -list_sp6 69 numbers from one image, format: x1, y1, score1, x2, y2, score2, ...
    -F fundamental matrix from C4 to C6
    
    Returns:
    -total_distance total distance between epolar line and the joint that has the same name
    """
    total_distance = 0
    for counter in range (0,23):
        pt_c4 = np.asarray(list_sp4[3*counter:(3*counter + 3)])
        pt_c6 = np.asarray(list_sp6[3*counter:(3*counter + 3)])
        if pt_c4[2] > 0.1 and pt_c4[2] > 0.1:
            dis4, dis6 = get_distance(pt_c4, pt_c6, P4, P6)
            distance = np.abs(dis6)
            total_distance += distance        
    return total_distance

def getEveryoneSingleFrame(path_json4, path_json6, frame_num, filtered_index_C4, filtered_index_C6):
    """get a list of all the people in one frame
    
    Args:
    -path_json4 path of json for C4
    -path_json6 path of json for C6
    
    Returns:
    -list_kp_4 a list
    -list_kp_6 a list
    """
    list_kp_4 = []
    list_kp_6 = []
    with open(path_json4) as json_data:
        data_dict4 = json.load(json_data)
        L4 = len(data_dict4)
    with open(path_json6) as json_data:
        data_dict6 = json.load(json_data)
        L6 = len(data_dict6)
        
    for counter4 in range (L4):
        name = data_dict4[counter4]['image_id'] 
        anno_id = data_dict4[counter4]['id']
        index = int(name[:-4])
        if index == frame_num and anno_id in filtered_index_C4:
            list_kp_4.append(data_dict4[counter4]['keypoints'])
            
    for counter6 in range (L6):
        name = data_dict6[counter6]['image_id'] 
        anno_id = data_dict6[counter6]['id']
        index = int(name[:-4])
        if index == frame_num and anno_id in filtered_index_C6:
            list_kp_6.append(data_dict6[counter6]['keypoints'])           
    return list_kp_4,list_kp_6

def getTrace(M_index):
    """get the indexes of dictionary that we care about based on feet matching
    
    -Args:
    M_index: a N by M matrix of corresponding annotation id in json (2D), N corresponds C6, M corresponds C4
    
    -Returns:
    filtered_index_C4: a list that consists of all the anno indexes of C4 we care about
    filtered_index_C6: a list that consists of all the anno indexes of C6 we care about
    """
    interesting_num_C4 = (M_index[0].shape)[0]
    interesting_num_C6 = (M_index[:,0].shape)[0]

    filtered_index_C4, filtered_index_C6 = [], []

    for counter1 in range (interesting_num_C4):
        filtered_index_C4.append(int(M_index[0][counter1][5:9]))

    for counter2 in range (interesting_num_C6):
        filtered_index_C6.append(int(M_index[:,0][counter2][0:4]))
    return filtered_index_C4, filtered_index_C6
