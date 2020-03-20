'''
Created on Dec 14, 2018

@author: Zongyang
'''
import cv2
import os,sys,json, random, csv
import numpy as np
from matplotlib import cm
from glob import glob
import matplotlib.pyplot as plt
from datetime import date, timedelta
from plyfile import PlyData

PLOT_RANGE_NUM = 54
PLOT_COL_NUM = 32
HIST_BIN_NUM = 400

SEL_MAX = 0
SEL_RAN = 1


def main():
    
    '''
    pImg = cv2.imread('/Users/nijiang/Desktop/Scanner3D/2016-07-02/2016-07-01__02-12-49-960/b7d702de-2490-4a7a-8813-096cc0c843f3__Top-heading-west_0_p.png', -1)
    gImg = cv2.imread('/Users/nijiang/Desktop/Scanner3D/2016-07-02/2016-07-01__02-12-49-960/b7d702de-2490-4a7a-8813-096cc0c843f3__Top-heading-west_0_g.png', -1)
    plyFile = PlyData.read('/Users/nijiang/Desktop/Scanner3D/2016-07-02/2016-07-01__02-12-49-960/b7d702de-2490-4a7a-8813-096cc0c843f3__Top-heading-west_0.ply')
    plyData = plyFile.elements[0]
    
    rel = ply_down_sampling(pImg, gImg, plyData, 20, 0)
    
    write_ply('/Users/nijiang/Desktop/Scanner3D/downSample1.ply', rel)
    '''
    
    in_dir = '/media/zli/Elements/ua-mac/Level_2/s6_downsample/downSamplingMedial'
    out_dir = '/media/zli/Elements/ua-mac/Level_2/s6_downsample/V_quantile/99'
    
    create_plot_betyCsv(in_dir, out_dir, 3, 0, 0.99)
    
    
    return


def plotNum_to_fieldPartition(plotNum):
    "Converts plot number to field partition"
    plot_row = 0
    plot_col = 0
    cols = 32
    col = plotNum % cols
    if col == 0:
        plot_row = plotNum / cols
        if (plot_row % 2 == 0):
            plot_col = 1
        else:
            plot_col = 32
            
        return plot_row, plot_col
    
    
    plot_row = plotNum/cols +1
    plot_col = col
    if (plot_row % 2 == 0):
        plot_col = cols - col + 1
    
    return plot_row, plot_col

def plotNum_to_fieldPartition_864(plotNum):
    "Converts plot number to field partition"
    plot_row = 0
    plot_col = 0
    cols = 16
    col = plotNum % cols
    if col == 0:
        plot_row = plotNum / cols
        if (plot_row % 2 == 0):
            plot_col = 1
        else:
            plot_col = 16
            
        return plot_row, plot_col
    
    
    plot_row = plotNum/cols +1
    plot_col = col
    if (plot_row % 2 == 0):
        plot_col = cols - col + 1
    
    return plot_row, plot_col

def create_plot_height_traits_stereo(in_dir, out_dir, str_date):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    base_name = '{}_heightHist.npy'.format(str_date) #2017-06-30_heightHist.npy
    file_path = os.path.join(in_dir, base_name)
    if not os.path.isfile(file_path):
        return
    
    heightHist = np.load(file_path)
    out_hist = np.zeros((1728, 3))
    out_hist[:] = np.nan
    array_ind = 0
    save_flag = False
    for i in range(1728):
        if np.amax(heightHist[i])>0:
            save_item = np.zeros(3)
            plot_row, plot_col = plotNum_to_fieldPartition(i+1)
            save_item[0] = plot_row-1
            save_item[1] = plot_col-1
            targetHist = heightHist[i]
            targetHist = targetHist/np.sum(targetHist)
            quantiles = np.cumsum(targetHist)
            b = np.arange(len(quantiles))
            c = b[quantiles>0.98]
            save_item[2] = min(c)
            out_hist[array_ind] = save_item
            save_flag = True
        array_ind += 1
            
    if save_flag:
        out_csv_path = os.path.join(out_dir, '{}_heightHist.csv'.format(str_date))
        with open(out_csv_path, 'w') as f:
            for i in range(len(out_hist)):
                out_np_line = out_hist[i]
                if np.amax(out_np_line) > 0:
                    out_np_line = out_np_line.astype(int)
                    print_line = ','.join(map(str,out_np_line))
                    f.write(print_line+'\n')
                else:
                    print_line = ','.join(map(str,out_np_line))
                    f.write(print_line+'\n')
    
    return out_hist

def create_plot_betyCsv(in_dir, out_dir, medial_level, gridSize, para = 0.98, sensor_d='e'):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    out_sub_dir = os.path.join(out_dir, str(medial_level), str(gridSize))
    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)
    
    d1 = date(2018, 4, 25)  # start date
    d2 = date(2018, 8, 1)  # end date

    delta = d2 - d1         # timedelta
    
    (fields, traits) = get_traits_table_height_quantile()
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        base_name = '{}_heightHist_{}_{}.npy'.format(str_date, sensor_d, str(gridSize)) #2017-06-30_heightHist_e_0.npy
        file_path = os.path.join(in_dir, base_name)
        if not os.path.isfile(file_path):
            continue
        
        heightHist = np.load(file_path)
        heightHist = heightHist[medial_level]
        if np.amax(heightHist) == 0:
            continue
        nonZeroIndex = np.nonzero(heightHist)
        max_y_ind = np.amax(nonZeroIndex[2])
        out_hist = np.zeros((864, 3))
        out_hist[:] = np.nan
        array_ind = 0
        save_flag = False
        for x in range(54):
            for y in range(16):
                if np.amax(heightHist[x,y])>0:
                    if heightHist[x,y,max_y_ind] < 50:
                        save_item = np.zeros(3)
                        save_item[0] = x
                        save_item[1] = y
                        targetHist = heightHist[x,y]
                        targetHist = targetHist/np.sum(targetHist)
                        quantiles = np.cumsum(targetHist)
                        b = np.arange(len(quantiles))
                        c = b[quantiles>para]
                        save_item[2] = min(c)-5
                        out_hist[array_ind] = save_item
                        save_flag = True
                array_ind += 1
                
        if save_flag:
            out_csv_path = os.path.join(out_sub_dir, '{}_99th_quantile.csv'.format(str_date))
            csvHandle = open(out_csv_path, 'w')
            csvHandle.write(','.join(map(str, fields)) + '\n')
            for j in range(len(out_hist)):
                out_np_line = out_hist[j]
                if np.amax(out_np_line) > 0:
                    str_time = str_date+'T12:00:00'
                    traits['local_datetime'] = str_time
                    traits['99th_quantile_canopy_height'] = str(int(out_np_line[2]))
                    traits['site'] = parse_site_from_range_column(out_np_line[0], out_np_line[1], 6)
                    trait_list = generate_traits_list_height(traits)
                    csvHandle.write(','.join(map(str, trait_list)) + '\n')
    
            csvHandle.close()
    #submitToBety(out_file)
    #betydb.submit_traits(out_file, filetype='csv', betykey=betydb.get_bety_key(), betyurl=betydb.get_bety_url())

    return

def stereo_hist_to_bety_format(in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    d1 = date(2016, 10, 16)  # start date
    d2 = date(2016, 11, 7)  # end date

    delta = d2 - d1         # timedelta
    
    (fields, traits) = get_traits_table_stereo_height_quantile()
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        base_name = '{}_stereoHeight.npy'.format(str_date) #2016-10-16_stereoHeight.npy
        file_path = os.path.join(in_dir, base_name)
        if not os.path.isfile(file_path):
            continue
        
        heightHist = np.load(file_path)
        if np.amax(heightHist) == 0:
            continue
        out_hist = np.zeros((864, 3))
        out_hist[:] = np.nan
        array_ind = 0
        save_flag = False
        for i in range(864):
            if np.amax(heightHist[2*i])>0:
                save_item = np.zeros(3)
                plot_row, plot_col = plotNum_to_fieldPartition_864(i+1)
                save_item[0] = plot_row-1
                save_item[1] = plot_col-1
                targetHist = heightHist[2*i]
                targetHist = targetHist/np.sum(targetHist)
                quantiles = np.cumsum(targetHist)
                b = np.arange(len(quantiles))
                c = b[quantiles>0.98]
                save_item[2] = min(c)
                out_hist[array_ind] = save_item
                save_flag = True
            array_ind += 1
                
        if save_flag:
            out_csv_path = os.path.join(out_dir, '{}_98th_quantile_stereoHeight.csv'.format(str_date))
            csvHandle = open(out_csv_path, 'w')
            csvHandle.write(','.join(map(str, fields)) + '\n')
            for j in range(len(out_hist)):
                out_np_line = out_hist[j]
                if np.amax(out_np_line) > 0:
                    str_time = str_date+'T12:00:00'
                    traits['local_datetime'] = str_time
                    traits['98th_quantile_canopy_height'] = str(int(out_np_line[2]))
                    traits['site'] = parse_site_from_range_column(out_np_line[0], out_np_line[1], 2)
                    trait_list = generate_traits_list_height(traits)
                    csvHandle.write(','.join(map(str, trait_list)) + '\n')
    
            csvHandle.close()
    #submitToBety(out_file)
    #betydb.submit_traits(out_file, filetype='csv', betykey=betydb.get_bety_key(), betyurl=betydb.get_bety_url())

    return
    
    

def create_plot_height_quantile_remove_over_height(in_dir, out_dir, medial_level, gridSize, sensor_d='e'):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    out_sub_dir = os.path.join(out_dir, str(medial_level), str(gridSize))
    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)
    
    d1 = date(2018, 4, 15)  # start date
    d2 = date(2018, 7, 29)  # end date

    delta = d2 - d1         # timedelta
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        base_name = '{}_heightHist_{}_{}.npy'.format(str_date, sensor_d, str(gridSize)) #2017-06-30_heightHist_e_0.npy
        file_path = os.path.join(in_dir, base_name)
        if not os.path.isfile(file_path):
            continue
        
        heightHist = np.load(file_path)
        heightHist = heightHist[medial_level]
        if np.amax(heightHist) == 0:
            continue
        nonZeroIndex = np.nonzero(heightHist)
        max_y_ind = np.amax(nonZeroIndex[2])
        out_hist = np.zeros((864, 3))
        out_hist[:] = np.nan
        array_ind = 0
        save_flag = False
        for x in range(54):
            for y in range(16):
                if np.amax(heightHist[x,y])>0:
                    if heightHist[x,y,max_y_ind] < 10:
                        save_item = np.zeros(3)
                        save_item[0] = x
                        save_item[1] = y
                        targetHist = heightHist[x,y]
                        targetHist = targetHist/np.sum(targetHist)
                        quantiles = np.cumsum(targetHist)
                        b = np.arange(len(quantiles))
                        c = b[quantiles>0.98]
                        save_item[2] = min(c)
                        out_hist[array_ind] = save_item
                        save_flag = True
                array_ind += 1
                
        if save_flag:
            out_csv_path = os.path.join(out_sub_dir, '{}_heightHist.csv'.format(str_date))
            with open(out_csv_path, 'w') as f:
                for i in range(len(out_hist)):
                    out_np_line = out_hist[i]
                    if np.amax(out_np_line) > 0:
                        out_np_line = out_np_line.astype(int)
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
                    else:
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
    
    return

def ply_down_sampling(pImg, gImg, plyData, gridSize, methodFlag=SEL_MAX):
    
    # no down-sampling in this case
    if gridSize == 0:
        return plyData.data
    
    # get relationship between ply files and png files, that means each point in the ply file 
    # should have a corresponding pixel in png files, both depth png and gray png
    pHei, pWid = pImg.shape[:2]
    gHei, gWid = gImg.shape[:2]
    if pWid == gWid:
        gPix = np.array(gImg).ravel()
        gIndex = (np.where(gPix>32))
        tInd = gIndex[0]
    else:
        pPix = np.array(pImg)
        pPix = pPix[:, 2:].ravel()
        pIndex = (np.where(pPix != 0))
        
        gPix = np.array(gImg).ravel()
        gIndex = (np.where(gPix>33))
        tInd = np.intersect1d(gIndex[0], pIndex[0])
                          
    nonZeroSize = tInd.size
    
    pointSize = plyData.count
    
    # if point size do not match, return
    if nonZeroSize != pointSize:
        return
    
    # Initial data structures
    gIndexImage = np.zeros(gWid*gHei)
    
    gIndexImage[tInd] = np.arange(1,pointSize+1)
    
    gIndexImage_ = np.reshape(gIndexImage, (-1, gWid))
    
    windowSize = gridSize
    xyScale = 1
    
    relPointData = []

    # move ROI in a window size to do the meta analysis
    for i in np.arange(0+windowSize*xyScale, gWid-windowSize*xyScale, windowSize*xyScale*2):
        for j in np.arange(0+windowSize, gHei-windowSize, windowSize*2):
            if methodFlag == SEL_MAX:
                plyIndices = gIndexImage_[j-windowSize:j+windowSize+1, i-windowSize*xyScale:i+windowSize*xyScale+1]
                plyIndices = plyIndices.ravel()
                plyIndices_ = np.where(plyIndices>0)
                localIndex = plyIndices[plyIndices_[0]].astype('int64')
                localP = plyData.data[localIndex-1]
                if len(localP) == 0:
                    continue
                
                maxZ = localP[np.argmax(localP["z"])]
                relPointData.append(maxZ)
                
            if methodFlag == SEL_RAN:
                nCount = 0
                xRange = [j-windowSize, j+windowSize+1]
                yRange = [i-windowSize*xyScale, i+windowSize*xyScale+1]
                
                
                
                selInd = 0
                while nCount < 10:
                    nCount += 1
                    xInd = random.randint(xRange[0], xRange[1])
                    yInd = random.randint(yRange[0], yRange[1])
                    if gIndexImage_[xInd, yInd] != 0:
                        selInd = gIndexImage_[xInd, yInd].astype('int64')
                        break
                    
                if selInd != 0:
                    relPointData.append(plyData.data[selInd-1])
    
    relPointData = np.asarray(relPointData)
    
    return relPointData

def save_sub_ply(subData, src, outFile):
    
    src.elements[0].data = subData
    src.elements[0].count = len(subData)
    src.write(outFile)
    
    return

def create_plot_height_histogram_remove_over_height(in_dir, out_dir, medial_level, gridSize, sensor_d='e'):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    out_sub_dir = os.path.join(out_dir, str(medial_level), str(gridSize))
    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)
    
    d1 = date(2018, 7, 7)  # start date
    d2 = date(2018, 7, 20)  # end date

    delta = d2 - d1         # timedelta
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        base_name = '{}_heightHist_{}_{}.npy'.format(str_date, sensor_d, str(gridSize)) #2017-06-30_heightHist_e_0.npy
        file_path = os.path.join(in_dir, base_name)
        if not os.path.isfile(file_path):
            continue
        
        heightHist = np.load(file_path)
        heightHist = heightHist[medial_level]
        if np.amax(heightHist) == 0:
            continue
        nonZeroIndex = np.nonzero(heightHist)
        max_y_ind = np.amax(nonZeroIndex[2])
        out_hist = np.zeros((864, 502))
        out_hist[:] = np.nan
        array_ind = 0
        save_flag = False
        for x in range(54):
            for y in range(16):
                if np.amax(heightHist[x,y])>0:
                    if heightHist[x,y,max_y_ind] < 10:
                        save_item = np.zeros(502)
                        save_item[0] = x
                        save_item[1] = y
                        save_item[2:] = heightHist[x, y]
                        out_hist[array_ind] = save_item
                        save_flag = True
                    
                array_ind += 1
                
        if save_flag:
            out_csv_path = os.path.join(out_sub_dir, '{}_heightHist_{}_{}.csv'.format(str_date, sensor_d, str(gridSize)))
            with open(out_csv_path, 'w') as f:
                for i in range(len(out_hist)):
                    out_np_line = out_hist[i][:-100]
                    if np.amax(out_np_line) > 0:
                        out_np_line = out_np_line.astype(int)
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
                    else:
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
    
    
    return

def create_plot_height_histogram(in_dir, out_dir, medial_level, gridSize, sensor_d='e'):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    out_sub_dir = os.path.join(out_dir, str(medial_level), str(gridSize))
    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)
    
    d1 = date(2018, 4, 15)  # start date
    d2 = date(2018, 7, 29)  # end date

    delta = d2 - d1         # timedelta
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        base_name = '{}_heightHist_{}_{}.npy'.format(str_date, sensor_d, str(gridSize)) #2017-06-30_heightHist_e_0.npy
        file_path = os.path.join(in_dir, base_name)
        if not os.path.isfile(file_path):
            continue
        
        heightHist = np.load(file_path)
        heightHist = heightHist[medial_level]
        out_hist = np.zeros((864, 502))
        out_hist[:] = np.nan
        array_ind = 0
        save_flag = False
        for x in range(54):
            for y in range(16):
                if np.amax(heightHist[x,y])>0:
                    save_item = np.zeros(502)
                    save_item[0] = x
                    save_item[1] = y
                    save_item[2:] = heightHist[x, y]
                    out_hist[array_ind] = save_item
                    save_flag = True
                    
                array_ind += 1
                
        if save_flag:
            out_csv_path = os.path.join(out_sub_dir, '{}_heightHist_{}_{}.csv'.format(str_date, sensor_d, str(gridSize)))
            with open(out_csv_path, 'w') as f:
                for i in range(len(out_hist)):
                    out_np_line = out_hist[i][:-100]
                    if np.amax(out_np_line) > 0:
                        out_np_line = out_np_line.astype(int)
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
                    else:
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
    
    
    return

def create_sub_plot_height_histogram(in_dir, out_dir, gridSize, sensor_d='e'):
    
    out_dir = os.path.join(out_dir, str(gridSize))
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    d1 = date(2017, 6, 1)  # start date
    d2 = date(2017, 6, 7)  # end date

    delta = d2 - d1         # timedelta
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        base_name = '{}_heightHist_subplot_{}_{}.npy'.format(str_date, sensor_d, str(gridSize)) #2017-04-27_heightHist_subplot_e_0.npy
        file_path = os.path.join(in_dir, base_name)
        if not os.path.isfile(file_path):
            continue
        
        heightHist = np.load(file_path)
        out_hist = np.zeros((1728, 502))
        out_hist[:] = np.nan
        array_ind = 0
        save_flag = False
        for x in range(54):
            for y in range(32):
                if np.amax(heightHist[x,y])>0:
                    save_item = np.zeros(502)
                    save_item[0] = x
                    save_item[1] = y
                    save_item[2:] = heightHist[x, y]
                    out_hist[array_ind] = save_item
                    save_flag = True
                    
                array_ind += 1
        
        if save_flag>0:
            out_csv_path = os.path.join(out_dir, '{}_heightHist_subplot_{}_{}.csv'.format(str_date, sensor_d, str(gridSize)))
            with open(out_csv_path, 'w') as f:
                for i in range(len(out_hist)):
                    out_np_line = out_hist[i][:-100]
                    if np.amax(out_np_line) > 0:
                        out_np_line = out_np_line.astype(int)
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
                    else:
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
    
    return

def create_plot_height_traits(in_dir, out_dir, gridSize, sensor_d='e'):
    
    out_dir = os.path.join(out_dir, str(gridSize))
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    d1 = date(2018, 5, 1)  # start date
    d2 = date(2018, 8, 1)  # end date

    delta = d2 - d1         # timedelta
    
    param_percentile = 0.95
    
    for i in range(delta.days + 1):
        str_date = str(d1 + timedelta(days=i))
        base_name = '{}_heightHist_subplot_{}_{}.npy'.format(str_date, sensor_d, str(gridSize)) #2017-04-27_heightHist_subplot_e_0.npy
        file_path = os.path.join(in_dir, base_name)
        if not os.path.isfile(file_path):
            continue
        
        heightHist = np.load(file_path)
        out_hist = np.zeros((864, 3))
        out_hist[:] = np.nan
        array_ind = 0
        save_flag = False
        for x in range(54):
            for y in range(16):
                if np.amax(heightHist[x,y])>0:
                    save_item = np.zeros(3)
                    save_item[0] = x
                    save_item[1] = y
                    
                    targetHist = heightHist[x, y]
                    targetHist = targetHist/np.sum(targetHist)
                    quantiles = np.cumsum(targetHist)
                    b = np.arange(len(quantiles))
                    c = b[quantiles>param_percentile]
                    save_item[2] = min(c)
                    out_hist[array_ind] = save_item
                    save_flag = True
                    
                array_ind += 1
        
        if save_flag>0:
            out_csv_path = os.path.join(out_dir, '{}_heightHist_plot_95th_Height.csv'.format(str_date))
            with open(out_csv_path, 'w') as f:
                for i in range(len(out_hist)):
                    out_np_line = out_hist[i]
                    if np.amax(out_np_line) > 0:
                        out_np_line = out_np_line.astype(int)
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
                    else:
                        print_line = ','.join(map(str,out_np_line))
                        f.write(print_line+'\n')
    
    return

def offset_choice(scanDirection, sensor_d):
    
    if sensor_d == 'w':
        if scanDirection == 'True':
            ret = -3.60#-3.08
        else:
            ret = -25.711#-25.18
            
    if sensor_d == 'e':
        if scanDirection == 'True':
            ret = -3.60#-3.08
        else:
            ret = -25.711#-25.18
    
    return ret

def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))
    
    
def lower_keys(in_dict):
    if type(in_dict) is dict:
        out_dict = {}
        for key, item in in_dict.items():
            out_dict[key.lower()] = lower_keys(item)
        return out_dict
    elif type(in_dict) is list:
        return [lower_keys(obj) for obj in in_dict]
    else:
        return in_dict
    
    
def find_input_files(ply_path, json_path):
    metadata_suffix = '_metadata.json'
    metas = [os.path.basename(meta) for meta in glob(os.path.join(json_path, '*' + metadata_suffix))]
    if len(metas) == 0:
        fail('No metadata file found in input directory.')

    ply_suffix = 'Top-heading-west_0.ply'
    plyWests = [os.path.basename(ply) for ply in glob(os.path.join(ply_path, '*' + ply_suffix))]
    if len(plyWests) == 0:
        fail('No west file found in input directory.')
        
    ply_suffix = 'Top-heading-east_0.ply'
    plyEasts = [os.path.basename(ply) for ply in glob(os.path.join(ply_path, '*' + ply_suffix))]
    if len(plyEasts) == 0:
        fail('No east file found in input directory.')

    return metas, plyWests, plyEasts

def find_input_files_all(ply_path, json_path):
    metadata_suffix = '_metadata.json'
    metas = [os.path.basename(meta) for meta in glob(os.path.join(json_path, '*' + metadata_suffix))]
    if len(metas) == 0:
        fail('No metadata file found in input directory.')
        
    png_suffix = 'Top-heading-west_0_g.png'
    gImgWests = [os.path.basename(meta) for meta in glob(os.path.join(json_path, '*' + png_suffix))]
    if len(gImgWests) == 0:
        fail('No gImgWests file found in input directory.')
        
    png_suffix = 'Top-heading-west_0_p.png'
    pImgWests = [os.path.basename(meta) for meta in glob(os.path.join(json_path, '*' + png_suffix))]
    if len(pImgWests) == 0:
        fail('No pImgWests file found in input directory.')
        
    png_suffix = 'Top-heading-east_0_g.png'
    gImgEasts = [os.path.basename(meta) for meta in glob(os.path.join(json_path, '*' + png_suffix))]
    if len(gImgEasts) == 0:
        fail('No gImgEasts file found in input directory.')
        
    png_suffix = 'Top-heading-east_0_p.png'
    pImgEasts = [os.path.basename(meta) for meta in glob(os.path.join(json_path, '*' + png_suffix))]
    if len(pImgEasts) == 0:
        fail('No pImgEasts file found in input directory.')

    ply_suffix = 'Top-heading-west_0.ply'
    plyWests = [os.path.basename(ply) for ply in glob(os.path.join(ply_path, '*' + ply_suffix))]
    if len(plyWests) == 0:
        fail('No west file found in input directory.')
        
    ply_suffix = 'Top-heading-east_0.ply'
    plyEasts = [os.path.basename(ply) for ply in glob(os.path.join(ply_path, '*' + ply_suffix))]
    if len(plyEasts) == 0:
        fail('No east file found in input directory.')

    return metas, plyWests, plyEasts, gImgWests, pImgWests, gImgEasts, pImgEasts

def get_position(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        sensor_fix_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        camera_x = '2.070'#sensor_fix_meta['scanner west location in camera box x [m]']
        camera_z = '1.135'
        

    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])

    try:
        x = float(gantry_x) + float(camera_x)
        y = float(gantry_y)
        z = float(gantry_z) + float(camera_z)
    except ValueError as err:
        fail('Corrupt positions, ' + err.args[0])
    return (x, y, z)


def get_direction(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        scan_direction = gantry_meta["scanisinpositivedirection"]
        
    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        
    return scan_direction

def load_histogram_from_npy(in_dir, sensor_d):
    
    file_path = os.path.join(in_dir, 'heightHist_' + sensor_d + '.npy')
    hist = np.load(file_path)
    
    return hist

def load_histogram_from_both_npy(in_dir):
    
    file_path_e = os.path.join(in_dir, 'heightHist_e.npy')
    if not os.path.exists(file_path_e):
        return []
    hist_e = np.load(file_path_e)
    
    file_path_w = os.path.join(in_dir, 'heightHist_w.npy')
    if not os.path.exists(file_path_w):
        return []
    hist_w = np.load(file_path_w)
    
    return hist_e, hist_w

def get_traits_table_height():
    
    fields = ('local_datetime', 'canopy_height', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'local_datetime' : '',
              'canopy_height' : [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': 'ZongyangLi',
              'citation_year': '2017',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': 'Scanner 3d ply data to height'}

    return (fields, traits)

def generate_traits_list_height(traits):
    # compose the summary traits
    trait_list = [  traits['local_datetime'],
                    traits['90th_quantile_canopy_height'],
                    traits['access_level'],
                    traits['species'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method']
                ]

    return trait_list

def get_traits_table_height_quantile():
    
    fields = ('local_datetime', '90th_quantile_canopy_height', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'local_datetime' : '',
              '90th_quantile_canopy_height' : [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': 'ZongyangLi',
              'citation_year': '2018',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': 'Scanner 3d ply data to 90th quantile height'}

    return (fields, traits)

def get_traits_table_stereo_height_quantile():

    fields = ('local_datetime', '98th_quantile_canopy_height', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'local_datetime' : '',
              '98th_quantile_canopy_height' : [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': 'ZongyangLi',
              'citation_year': '2019',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': 'Stereo RGB data to 98th quantile height'}

    return (fields, traits)


def field_x_2_range(x_position, convt):
    
    xRange = 0
    
    for i in range(convt.max_range):
        xmin = convt.np_bounds[i][0][0]
        xmax = convt.np_bounds[i][0][1]
        if (x_position > xmin) and (x_position <= xmax):
            xRange = i + 1
    
    return xRange

def field_2_plot(x_position, y_row, convt):

    xRange = 0
    for i in range(PLOT_RANGE_NUM):
        xmin = convt.np_bounds_subplot[i][0][0]
        xmax = convt.np_bounds_subplot[i][0][1]
        if (x_position > xmin) and (x_position <= xmax):
            xRange = i + 1
            
            plotNum = convt.fieldPartition_to_plotNum_32(xRange, y_row)
            
            return plotNum
    
    return 0

def plotNum_to_range_col_1728(plotNum):
    
    
    
    
    
    
    
    return

def parse_site_from_plotNum_1728(plotNum, convt):

    plot_row = 0
    plot_col = 0
    
    cols = 32
    col = (plotNum-1) % cols + 1
    row = (plotNum-1) / cols + 1
    
    
    if (row % 2) != 0:
        plot_col = col
    else:
        plot_col = cols - col + 1
    
    Range = row
    Column = (plot_col + 1) / 2
    if (plot_col % 2) != 0:
        subplot = 'W'
    else:
        subplot = 'E'
        
    seasonNum = convt.seasonNum
        
    rel = 'MAC Field Scanner Season {} Range {} Column {} {}'.format(str(seasonNum), str(Range), str(Column), subplot)
    
    return rel

def parse_site_from_range_column(row, col, seasonNum):
    
    rel = 'MAC Field Scanner Season {} Range {} Column {}'.format(str(seasonNum), str(int(row+1)), str(int(col+1)))
    
    return rel

def find_result_file_s6(in_dir, sensor_d, gridSize):
    
    metadata_suffix = os.path.join(in_dir, '*_metadata.json')
    metas = glob(metadata_suffix)
    if len(metas) == 0:
        return [], [], []

    hist_file = os.path.join(in_dir, 'hist_'+sensor_d+'_'+str(gridSize*2)+'.npy')
    if os.path.isfile(hist_file) == False:
        return [], [], []
    
    hist_subplot_file = os.path.join(in_dir, 'hist_subplot_'+sensor_d+'_'+str(gridSize*2)+'.npy')
    if os.path.isfile(hist_subplot_file) == False:
        return [], [], []
    
    return metas[0], hist_file, hist_subplot_file

def find_result_file_s4(in_dir, sensor_d):
    
    metadata_suffix = os.path.join(in_dir, '*_metadata.json')
    metas = glob(metadata_suffix)
    if len(metas) == 0:
        return [], [], []

    hist_file = os.path.join(in_dir, 'hist_'+sensor_d+'.npy')
    if os.path.isfile(hist_file) == False:
        return [], [], []
    
    hist_subplot_file = os.path.join(in_dir, 'hist_subplot_'+sensor_d+'.npy')
    if os.path.isfile(hist_subplot_file) == False:
        return [], [], []

    return metas[0], hist_file, hist_subplot_file

def find_result_files(in_dir, sensor_d):
    
    metadata_suffix = os.path.join(in_dir, '*_metadata.json')
    metas = glob(metadata_suffix)
    if len(metas) == 0:
        #fail('No metadata file found in input directory.')
        return [], []

    hist_file = os.path.join(in_dir, 'hist_'+sensor_d+'.npy')
    if os.path.isfile(hist_file) == False:
        #fail('No hist file or top file in input directory')
        return [], []

    return metas[0], hist_file

def save_points(ply_data, out_file, id):
    
    X = ply_data["x"]
    Y = ply_data["y"]
    Z = ply_data["z"]
    data_size = X.size
    
    index = (np.linspace(0,data_size-1,10000)).astype('int')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if data_size < 10:
        plt.savefig(out_file)
        plt.close()
        return
    
    colors = cm.rainbow(np.linspace(0, 1, 32))
    X = X[index]
    Y = Y[index]
    Z = Z[index]
    ax.scatter(X,Y,Z,color=colors[id], s=2)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.view_init(10, 0)
    plt.draw()
    plt.savefig(out_file)
    plt.close()
    
    return

def fail(reason):
    print >> sys.stderr, reason
    
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property float pix
end_header
'''

def write_ply(fn, verts):
    verts = np.asarray(verts)
    #verts = verts.reshape(-1, 3)
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d')
    
    return

if __name__ == "__main__":

    main()