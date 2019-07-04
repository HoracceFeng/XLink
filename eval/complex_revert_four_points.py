import numpy as np

def inSegment(p,line,line2):
    if line[0][0] == line[1][0]: # if line verticle
        if  p[1] > min(line[0][1],line[1][1]) and p[1] < max(line[0][1],line[1][1]):
            #if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]):
            if p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
                return True
    elif line[0][1] == line[1][1]: # if line horizontal
        if p[0] > min(line[0][0],line[1][0]) and p[0] < max(line[0][0],line[1][0]):
            #if p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
            if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]):
                return True
    else:
        if p[0] > min(line[0][0],line[1][0]) and p[0] < max(line[0][0],line[1][0]):
            # if line is not vertivle/honrizon, line2 may be verticle or horinzontal and thus x/y needed be check 
            if p[1] >= min(line2[0][1],line2[1][1]) and p[1] <= max(line2[0][1],line2[1][1]) and p[0] >= min(line2[0][0],line2[1][0]) and p[0] <= max(line2[0][0],line2[1][0]):
                return True
    return False

def getLinePara(line):
    a = line[0][1] - line[1][1]
    b = line[1][0] - line[0][0]
    c = line[0][0] *line[1][1] - line[1][0] * line[0][1]
    return a,b,c

def getCrossPoint(line1,line2):
    a1,b1,c1 = getLinePara(line1)
    a2,b2,c2 = getLinePara(line2)
    d = a1* b2 - a2 * b1
    p = [0,0]
    if d == 0: # d=0 when line1 line2 parallel
        return []
    else:
        p[0] = round((b1 * c2 - b2 * c1)*1.0 / d,2)
        p[1] = round((c1 * a2 - c2 * a1)*1.0 / d,2)
    #p = tuple(p)
    if inSegment(p,line1,line2):
        #print(p)
        return p
    else:
        return []

def fourpoint(coordinate):
    points = []
    for i in range(4):
        points.append([coordinate[i * 2], coordinate[i * 2 + 1]])
    
    cross_point = getCrossPoint([points[0],points[1]],[points[2],points[3]])
    if len(cross_point) == 0:
        cross_point = getCrossPoint([points[0],points[2]],[points[1],points[3]])
        if len(cross_point) == 0:
            cross_point = getCrossPoint([points[0],points[3]],[points[1],points[2]])
    
    left = []
    right = []
    points=np.array(points)

    #center_x = (max(points[:,0])+min(points[:,0]))/2
    #center_y = (max(points[:,1])+min(points[:,1]))/2
    for point in points:
        if point[0]<cross_point[0]:
            left.append(point)
        else:
            right.append(point)
    if len(left) ==2 and len(right)==2:
        if left[0][1]< left[1][1]:
            left_top = left[0]
            left_down = left[1]
        else:
            left_top = left[1]
            left_down = left[0]
        if right[0][1]< right[1][1]:
            right_top = right[0]
            right_down = right[1]
        else:
            right_top = right[1]
            right_down = right[0]
        return right_down,left_down,left_top,right_top
    else:
        print("the poi result is unnormal, program stop!")
        exit()