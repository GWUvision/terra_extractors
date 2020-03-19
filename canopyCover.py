'''
Created on Oct 31, 2016

@author: Zongyang
'''
import os, sys, json, argparse, multiprocessing
from glob import glob
from PIL import Image, ImageFilter
from scipy.ndimage.filters import convolve
import numpy as np
import terra_common
#import matplotlib.pyplot as plt
from datetime import date
import shutil
#import ganEnhancement
from datetime import date, timedelta,datetime
from skimage import morphology
import matplotlib.pyplot as plt
import cv2

#model = ganEnhancement.init_model()
CPUS = 40
SATUTATE_THRESHOLD = 245
MAX_PIXEL_VAL = 255
SMALL_AREA_THRESHOLD = 200

def options():
    
    parser = argparse.ArgumentParser(description='Canopy Cover Percent Extractor on Roger',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-i", "--in_dir", help="input directory")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-c", "--csv_dir", help="out csv directory")

    args = parser.parse_args()

    return args

def full_season_cc_frame(raw_rgb_dir, out_dir, start_date, end_date, convt):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    print(deltaDay.days)
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        raw_path = os.path.join(raw_rgb_dir, str_date)
        
        out_path = os.path.join(out_dir, str_date)
        
        if not os.path.isdir(raw_path):
            continue
        
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        
        #crop_rgb_imageToPlot(raw_path, out_path, plot_dir, convt)
        full_day_multi_process(raw_path, out_path, convt)
        #full_day_gen_cc(raw_path, out_path, convt)
    
    return

def full_day_multi_process(in_dir, out_path, convt):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    list_dirs = [os.path.join(in_dir,o) for o in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,o))]
    out_dirs = [os.path.join(out_path,o) for o in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,o))]
    numDirs = len(list_dirs)
    
    print ("Starting bin to cc conversion...")
    pool = multiprocessing.Pool()
    NUM_THREADS = min(CPUS,numDirs)
    print('numDirs:{}   NUM_THREADS:{}'.format(numDirs, NUM_THREADS))
    for cpu in range(NUM_THREADS):
        pool.apply_async(bin_to_png, [list_dirs[cpu::NUM_THREADS], out_dirs[cpu::NUM_THREADS], convt])
    pool.close()
    pool.join()
    print ("Completed bin to png conversion...")
    
    return

def bin_to_png(in_dirs, out_dirs, convt):
    for i, o in zip(in_dirs, out_dirs):
        try:
            gen_cc(i, o, convt)
            #bin_to_geotiff.stereo_test(s, s)
        except Exception as ex:
            fail("\tFailed to process folder %s: %s" % (i, str(ex)))


def process_all_data(in_dir, out_dir):
    
    list_dirs = os.listdir( in_dir )
    
    for dir in list_dirs:
        in_path = os.path.join(in_dir, dir)
        out_path = os.path.join(out_dir, dir)
        if not os.path.isdir(in_path):
            continue
        
        try:
            full_day_gen_cc(in_path, out_path)
        except Exception as ex:
            fail(in_path + str(ex))
    
    return


def full_day_gen_cc(in_dir, out_dir, convt):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    list_dirs = os.walk(in_dir)
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            #print("Start processing "+ d)
            i_path = os.path.join(in_dir, d)
            o_path = os.path.join(out_dir, d)
            if not os.path.isdir(i_path):
                continue
            
            gen_cc(i_path, o_path, convt)
    
    return

def full_day_gen_cc_from_image(in_dir, out_dir):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    list_dirs = os.walk(in_dir)
    
    out_csv_file = os.path.join(out_dir, 'ccAuto.csv')
    csv_handle = open(out_csv_file, 'w')
    
    for root, dirs, files in list_dirs:
        for f in files:
            if not f.endswith('.jpg'):
                continue
            
            input_file = os.path.join(in_dir, f)
            out_color_file = os.path.join(out_dir, f)
            out_bin_file = os.path.join(out_dir, f[:-4]+'_mask.jpg')
            ratio, outBin, ColorImg = gen_cc_enhanced(input_file, 3)
            
            if ratio == None:
                continue
            cv2.imwrite(out_color_file, ColorImg)
            cv2.imwrite(out_bin_file, outBin)
            
            out_line = '{},{}\n'.format(f, ratio)
            csv_handle.write(out_line)
            
    csv_handle.close()
    
    return

def gen_cc_from_binImage(in_dir, out_dir):
    
    list_dirs = os.walk(in_dir)
    
    out_csv_file = os.path.join(out_dir, 'ccAuto.csv')
    csv_handle = open(out_csv_file, 'w')
    
    for root, dirs, files in list_dirs:
        for f in files:
            if not f.endswith('.png'):
                continue
            
            input_file = os.path.join(in_dir, f)
            binMask = cv2.imread(input_file,0)
            c = np.count_nonzero(binMask)
            ratio = c/float(binMask.size)
            
            out_line = '{},{}\n'.format(f, ratio)
            csv_handle.write(out_line)
    
    csv_handle.close()
    return

def modify_param_process(in_dir, out_dir, csv_dir):
    
    spe_date = {'2017-05-17', '2017-05-24'}
    for str_date in spe_date:
        print(str_date)
        in_path = os.path.join(in_dir, str_date)
        out_path = os.path.join(out_dir, str_date)
        if not os.path.isdir(in_path):
            continue
        
        #integrate_cc_results(out_dir, str_date, csv_dir)
        
        convt = terra_common.CoordinateConverter()
        try:
            #q_flag = convt.bety_query(str_date, False)
            #if not q_flag:
            #    print('Bety query failed')
            #    continue
            #full_day_gen_cc(in_path, out_path, convt)
    
            integrate_cc_results(out_dir, str_date, csv_dir)
        except Exception as ex:
            fail(str_date + str(ex))
    
    
    return

def full_season_cc_integrate(cc_dir, out_dir, start_date, end_date, convt):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        #out_path = os.path.join(out_dir, str_date)
        
        integrate_cc_results(cc_dir, str_date, out_dir, convt)
    
    return

def copy_csv_to_outdir(in_dir, out_dir, start_date, end_date):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        in_path = os.path.join(in_dir, str_date)
        
        file_name = '{}CC_Bety.csv'.format(str_date)
        src_file_path = os.path.join(in_path, file_name)
        dst_file_path = os.path.join(out_dir, file_name)
        
        if not os.path.isfile(src_file_path):
            continue
        
        shutil.copyfile(src_file_path, dst_file_path)
    
    
    return

def cc_plots(in_dir, out_dir, start_date, end_date):
    
    # load npy data into super list
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    super_list = []
    date_list = []
    
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        in_path = os.path.join(in_dir, str_date)
        if not os.path.isdir(in_path):
            continue
        file_path = os.path.join(in_path, '{}_nparray.npy'.format(str_date))
        if not os.path.isfile(file_path):
            continue
        
        one_day_list = np.load(file_path)
        if np.amax(one_day_list)==0:
            continue
        super_list.append(one_day_list)
        date_list.append(str_date)
    
    # draw box plots
    for i in range(1, 865):
        plotNum = i
        box_plot(super_list, plotNum)
    
    
    return

def box_plot(super_list, plotNum):
   
    day_length = len(super_list)
     
    data = []
     
    for i in range(day_length):
        data.append(super_list[i][plotNum-1])
     
    fig, ax = plt.subplots()
     
    ax.boxplot(data)
     
    plt_title = 'Plot Number:%d' % plotNum
    plt.xlabel('Day')
    plt.ylabel('Canopy Cover Percentage')
    plt.title(plt_title)
    ax.set_ylim(ymin=0, ymax=1)
    plt.xticks(fontsize = 5)
     
    out_file = 'cc_%d.png' % plotNum
    out_file = os.path.join('/media/zli/Elements/ua-mac/Level_3/canopy_cover/box_plot', out_file)
    plt.savefig(out_file)
    plt.close()
   
    return

def process_specified_data(in_dir, out_dir):
    
    for day in range(7, 15):
        target_date = date(2016, 11, day)
        str_date = target_date.isoformat()
        print(str_date)
        in_path = os.path.join(in_dir, str_date)
        out_path = os.path.join(out_dir, str_date)
        if not os.path.isdir(in_path):
            continue
        try:
            full_day_gen_cc(in_path, out_path)
    
            integrate_cc_results(out_dir, str_date)
        except Exception as ex:
            fail(str_date + str(ex))
    
    return

def gen_cc(in_dir, out_dir, convt):
    
    meta, im_left = find_input_files(in_dir)
    if meta == None or im_left == None:
        return
    
    metadata = lower_keys(load_json(meta))
    plot_row, plot_col = get_plot_range_column(metadata, convt)
    if plot_row == 0 or plot_col == 0:
        return
    
    cc = get_CC_from_bin(im_left)
    
    if cc < 0:
        return
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    txt_file = os.path.join(out_dir, 'result.txt')
    
    text_file = open(txt_file, "w")
    text_file.write("plot_row=%d\n" % plot_row)
    text_file.write("plot_col=%d\n" % plot_col)
    text_file.write("cc=%f" % cc)
    text_file.close()
    
    return

def get_plot_range_column(metadata, convt):
    
    center_position, hh = parse_metadata(metadata)
    if center_position == None:
        return 0, 0
    
    plot_row, plot_col = convt.fieldPosition_to_fieldPartition(center_position[0], center_position[1])
    
    if hh < 7 or hh > 18:
        plot_row = 0
        plot_col = 0
    
    return plot_row, plot_col

def load_one_day_cc_result(in_dir, convt):
    
    if not os.path.isdir(in_dir):
        return
    
    list_dirs = os.walk(in_dir)
    
    cc_lst = [[[] for i in range(convt.max_col)] for j in range(convt.max_range)]
    
    for root, dirs, files in list_dirs:
        for dir in dirs:
            dir_path = os.path.join(in_dir, dir)
            if not os.path.isdir(dir_path):
                continue
            
            cc_file = os.path.join(dir_path, 'result.txt')
            
            plot_row, plot_col, cc = get_result_from_file(cc_file)
            
            if plot_row == 0 or plot_col == 0:
                continue
            
            cc_lst[plot_row-1][plot_col-1].append(cc)
    
    return cc_lst


def get_result_from_file(file_path):
    
    plot_row = 0
    plot_col = 0
    cc = 0
    
    if not os.path.isfile(file_path):
        return plot_row, plot_col, cc
    
    text_file = open(file_path, 'r')
    while True:
        
        line = text_file.readline()
        fields = line.split('=')
        if fields[0] == 'plot_row':
            plot_row = int(fields[1])
            continue
        
        if fields[0] == 'plot_col':
            plot_col = int(fields[1])
            continue
            
        if fields[0] == 'cc':
            cc = float(fields[1])
            break
    text_file.close()
    
    return plot_row, plot_col, cc

def integrate_cc_results(in_dir, str_date, out_dir, convt):
    
    sub_path = os.path.join(in_dir, str_date)
    print('start loading cc result')
    one_day_list = load_one_day_cc_result(sub_path, convt)
    if one_day_list == None:
        return
    
    gen_BETY_csv(one_day_list, str_date, out_dir, convt)

    return

def gen_BETY_csv(cc_list, str_date, out_dir, convt):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    print('start creating integrated result')
    csv_path = os.path.join(out_dir, str_date+'CC_Bety.csv')
    csv_handle = open(csv_path, 'w')
    
    (fields, traits) = get_traits_table()
    csv_handle.write(','.join(map(str, fields)) + '\n')
    
    for plot_row in range(convt.max_range):
        for plot_col in range(convt.max_col):
            plotData = cc_list[plot_row][plot_col]
            dataNum = len(plotData)
            
            if dataNum == 0:
                continue
            
            ccAve = sum(plotData) / float(len(plotData))
            str_time = str_date+'T12:00:00'
            traits['local_datetime'] = str_time
            traits['canopy_cover'] = ccAve
            traits['site'] = parse_site_from_range_column(plot_row, plot_col, convt.seasonNum)
            trait_list = generate_traits_list(traits)
            csv_handle.write(','.join(map(str, trait_list)) + '\n')
            
    
    csv_handle.close()
    
    #npy_path = os.path.join(out_dir, str_date+'_nparray.npy')
    #np.save(npy_path, cc_list)
    
    return

def parse_site_from_range_column(row, col, seasonNum):
    
    rel = 'MAC Field Scanner Season {} Range {} Column {}'.format(str(seasonNum), str(int(row+1)), str(int(col+1)))
    
    return rel

# Utility functions for modularity between command line and extractors
###########################################
def get_traits_table():
    # Compiled traits table
    fields = ('local_datetime', 'canopy_cover', 'access_level', 'species', 'site',
              'citation_author', 'citation_year', 'citation_title', 'method')
    traits = {'local_datetime' : '',
              'canopy_cover' : [],
              'access_level': '2',
              'species': 'Sorghum bicolor',
              'site': [],
              'citation_author': '"Zongyang, Li"',
              'citation_year': '2016',
              'citation_title': 'Maricopa Field Station Data and Metadata',
              'method': 'Canopy Cover Estimation from RGB images'}

    return (fields, traits)

def generate_traits_list(traits):
    # compose the summary traits
    trait_list = [  traits['local_datetime'],
                    traits['canopy_cover'],
                    traits['access_level'],
                    traits['species'],
                    traits['site'],
                    traits['citation_author'],
                    traits['citation_year'],
                    traits['citation_title'],
                    traits['method']
                ]

    return trait_list

def generate_cc_csv(fname, fields, trait_list):
    """ Generate CSV called fname with fields and trait_list """
    csv = open(fname, 'w')
    csv.write(','.join(map(str, fields)) + '\n')
    csv.write(','.join(map(str, trait_list)) + '\n')
    csv.close()

    return fname

def find_input_files(in_dir):
    
    json_suffix = os.path.join(in_dir, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        terra_common.fail('Could not find .json file')
        return None, None
        
        
    bin_suffix = os.path.join(in_dir, '*left.bin')
    bins = glob(bin_suffix)
    if len(bins) == 0:
        terra_common.fail('Could not find .bin file')
        return None, None
    
    return jsons[0], bins[0]


def get_plot_num(meta):
    
    center_position, hh = parse_metadata(meta)
    if center_position == None:
        return 0
    
    convt = terra_common.CoordinateConverter()
    
    plot_row, plot_col = convt.fieldPosition_to_fieldPartition(center_position[0], center_position[1])
    
    plotNum = convt.fieldPartition_to_plotNum(plot_row, plot_col)
    if hh < 7 or hh > 18:
        plotNum = 0
    
    return plotNum

def parse_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        capture_time =gantry_meta["time"]
        if len(capture_time) == 19:
            hh = int(capture_time[11:13])
        else:
            hh = 0
        
        cam_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        cam_x = cam_meta["location in camera box x [m]"]
        cam_y = cam_meta["location in camera box y [m]"]

        
        if "location in camera box z [m]" in cam_meta: # this may not be in older data
            cam_z = cam_meta["location in camera box z [m]"]
        else:
            cam_z = 0

    except KeyError as err:
        terra_common.fail('Metadata file missing key: ' + err.args[0])
        return None, None
        
    position = [float(gantry_x), float(gantry_y), float(gantry_z)]
    center_position = [position[0]+float(cam_x), position[1]+float(cam_y), position[2]+float(cam_z)]
    
    return center_position, hh

def get_localdatetime(metadata):
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        localTime = gantry_meta["time"]
    except KeyError as err:
        terra_common.fail('Metadata file missing key: ' + err.args[0])
        
    return localTime


def get_CC_from_bin(file_path):
    
    image = process_image(file_path, [3296, 2472])
    if image == None:
        return -1
    
    cv2Image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ratio, outBin, ColorImg = gen_cc_enhanced_imageInput(cv2Image, 5)
    
    if ratio == None:
        return -1
    
    '''
    input_file = os.path.join(in_dir, f)
    out_color_file = os.path.join(out_dir, f)
    out_bin_file = os.path.join(out_dir, f[:-4]+'_mask.jpg')
    ratio, outBin, ColorImg = gen_cc_enhanced(input_file, 3)
    
    if ratio == None:
        continue
    cv2.imwrite(out_color_file, ColorImg)
    cv2.imwrite(out_bin_file, outBin)
    
    if ratio < 0.5:
        base_name = os.path.basename(file_path)[:-4]
        debug_dir = '/media/zli/data/cc_debug2'
        
        out_color_file = os.path.join(debug_dir, base_name+'_'+str(round(ratio, 2))+'.jpg')
        print(out_color_file)
        Image.fromarray(image).save(out_color_file)
        out_mask_file = os.path.join(debug_dir, base_name+'_'+str(round(ratio, 2))+'.png')
        cv2.imwrite(out_mask_file, ColorImg)
    '''
    return ratio


def gen_cc_for_img(img, kernelSize):
    
    #im = Image.fromarray(img)
    
    #r, g, b = im.split()
    
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    sub_img = (g.astype('int') - r.astype('int') -2) > 0 # normal: -2
    
    mask = np.zeros_like(b)
    
    mask[sub_img] = 255
    
    im = Image.fromarray(mask)
    blur = im.filter(ImageFilter.BLUR)
    pix = np.array(blur)
    #blur = cv2.blur(mask,(kernelSize,kernelSize))
    sub_mask = pix > 128
    
    c = np.count_nonzero(sub_mask)
    ratio = c/float(b.size)
    
    return ratio

# check how many percent of pix close to 255 or 0
def check_saturation(img):
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    m1 = grayImg > SATUTATE_THRESHOLD
    m2 = grayImg < 20 # 20 is a threshold to classify low pixel value
    
    over_rate = float(np.sum(m1))/float(grayImg.size)
    low_rate = float(np.sum(m2))/float(grayImg.size)
    
    return over_rate, low_rate

# gen average pixel value from grayscale image
def check_brightness(img):
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    aveValue = np.average(grayImg)
    
    return aveValue

def getImageQuality(imgfile):
    
    img = Image.open(imgfile)
    img = np.array(img)

    NRMAC = MAC(img, img, img)

    return NRMAC

def MAC(im1,im2, im): # main function: Multiscale Autocorrelation (MAC)
    h, v, c = im1.shape
    if c>1:
        im  = np.matrix.round(rgb2gray(im))
        im1 = np.matrix.round(rgb2gray(im1))
        im2 = np.matrix.round(rgb2gray(im2))
    # multiscale parameters
    scales = np.array([2, 3, 5])
    FM = np.zeros(len(scales))
    for s in range(len(scales)):
        im1[0: h-1,:] = im[1:h,:]
        im2[0: h-scales[s], :]= im[scales[s]:h,:]
        dif = im*(im1 - im2)
        FM[s] = np.mean(dif)
    NRMAC = np.mean(FM)
    return NRMAC

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def gen_cc_enhanced_imageInput(input_img, kernelSize):
    
    # calculate image scores
    over_rate, low_rate = check_saturation(input_img)
    
    aveValue = check_brightness(input_img)
    
    # if low score, return None
    if low_rate > 0.15 or aveValue < 30 or aveValue > 195:
        return None, None, None
    
    # saturated image process
    if over_rate > 0.15:
        #return None, None, None
        binMask = gen_saturated_mask(input_img)
    else:   # nomal image process
        binMask = gen_mask(input_img)
        
    c = np.count_nonzero(binMask)
    ratio = c/float(binMask.size)
    
    rgbMask = gen_rgb_mask(input_img, binMask)
    
    return ratio, binMask, rgbMask

def gen_plant_mask(colorImg, kernelSize=3, thre=1):
    
    r = colorImg[:,:,2]
    g = colorImg[:,:,1]
    b = colorImg[:,:,0]
    
    sub_img = (g.astype('int') - r.astype('int') -0) > thre # normal: 1
    
    mask = np.zeros_like(b)
    
    mask[sub_img] = MAX_PIXEL_VAL
    
    blur = cv2.blur(mask,(kernelSize,kernelSize))
    pix = np.array(blur)
    sub_mask = pix > 128
    
    mask_1 = np.zeros_like(b)
    mask_1[sub_mask] = MAX_PIXEL_VAL
    
    return mask_1

def remove_small_area_mask(maskImg, min_area_size):
    
    mask_array = maskImg > 0
    rel_array = morphology.remove_small_objects(mask_array, min_area_size)
    
    rel_img = np.zeros_like(maskImg)
    rel_img[rel_array] = MAX_PIXEL_VAL
    
    return rel_img

def remove_small_holes_mask(maskImg, max_hole_size):
    
    mask_array = maskImg > 0
    rel_array = morphology.remove_small_holes(mask_array, max_hole_size)
    rel_img = np.zeros_like(maskImg)
    rel_img[rel_array] = MAX_PIXEL_VAL
    
    return rel_img

# connected component analysis for over saturation pixels
def over_saturation_process(rgb_img, init_mask, threshold = SATUTATE_THRESHOLD):
    
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    mask_over = gray_img > threshold
    
    mask_0 = gray_img < threshold
    
    src_mask_array = init_mask > 0
    
    mask_1 = src_mask_array & mask_0
    
    mask_1 = morphology.remove_small_objects(mask_1, SMALL_AREA_THRESHOLD)
    
    mask_over = morphology.remove_small_objects(mask_over, SMALL_AREA_THRESHOLD)
    
    rel_mask = saturated_pixel_classification(gray_img, mask_1, mask_over, 1)
    rel_img = np.zeros_like(gray_img)
    rel_img[rel_mask] = MAX_PIXEL_VAL
    
    return rel_img

# add saturated area into basic mask
def saturated_pixel_classification(gray_img, baseMask, saturatedMask, dilateSize=0):
    
    saturatedMask = morphology.binary_dilation(saturatedMask, morphology.diamond(dilateSize))
    
    rel_img = np.zeros_like(gray_img)
    rel_img[saturatedMask] = MAX_PIXEL_VAL
    
    label_img, num = morphology.label(rel_img, connectivity=2, return_num=True)
    
    rel_mask = baseMask
    
    for i in range(1, num):
        x = (label_img == i)
        
        if np.sum(x) > 100000: # if the area is too large, do not add it into basic mask
            continue
        
        if not (x & baseMask).any():
            continue
        
        rel_mask = rel_mask | x
    
    return rel_mask

def gen_saturated_mask(img):
    
    binMask = gen_plant_mask(img)
    binMask = remove_small_area_mask(binMask, 500)
    binMask = remove_small_holes_mask(binMask, 300)
    
    binMask = over_saturation_process(img, binMask, 245)
    
    binMask = remove_small_holes_mask(binMask, 4000)
    
    return binMask

def gen_mask(img):
    
    binMask = gen_plant_mask(img)
    binMask = remove_small_area_mask(binMask, 200)
    binMask = remove_small_holes_mask(binMask, 3000)
    
    return binMask

def gen_rgb_mask(img, binMask):
    
    rgbMask = cv2.bitwise_and(img, img, mask = binMask)
    
    return rgbMask

def process_image(im_path, shape):
    
    try:
        im = np.fromfile(im_path, dtype='uint8').reshape(shape[::-1])
        im_color = demosaic(im)
        im_color = np.rot90(im_color)
    except Exception as ex:
        print('Can not convert file from bin to RGB: {}'.format(im_path))
        return
    return im_color

def demosaic(im):
    # Assuming GBRG ordering.
    B = np.zeros_like(im)
    R = np.zeros_like(im)
    G = np.zeros_like(im)
    R[0::2, 1::2] = im[0::2, 1::2]
    B[1::2, 0::2] = im[1::2, 0::2]
    G[0::2, 0::2] = im[0::2, 0::2]
    G[1::2, 1::2] = im[1::2, 1::2]

    fG = np.asarray(
            [[0, 1, 0],
             [1, 4, 1],
             [0, 1, 0]]) / 4.0
    fRB = np.asarray(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]]) / 4.0

    im_color = np.zeros(im.shape+(3,), dtype='uint8') #RGB
    im_color[:, :, 0] = convolve(R, fRB)
    im_color[:, :, 1] = convolve(G, fG)
    im_color[:, :, 2] = convolve(B, fRB)
    return im_color

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

def fail(reason):
    print >> sys.stderr, reason
    
def main():
    
    print("start...")
    
    args = options()
    
    start_date = '2019-04-18'
    end_date = '2019-08-31'
    
    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query('2019-06-18') # All plot boundaries in one season should be the same, currently 2019-06-18 works best
    
    if not qFlag:
        return
    
    full_season_cc_frame(args.in_dir, args.out_dir, start_date, end_date, convt)
    
    full_season_cc_integrate(args.out_dir, args.csv_dir, start_date, end_date, convt)
    
    return

def test():
    
    
    im_left = '/media/zli/Elements/ua-mac/raw_data/stereoTop/2019-06-18/2019-06-18__10-39-39-399/d7f4fbf0-d476-48a6-b5b8-959ef8397998_left.bin'
    
    cc = get_CC_from_bin(im_left)
    '''
    in_dir = '/media/zli/Elements/ua-mac/Level_2/canopyCoverS4_integrate'
    out_dir = '/media/zli/Elements/ua-mac/Level_2/ccS4_plots/csv_dir'
    
    start_date = '2017-04-27'
    end_date = '2017-08-31'
    
    #full_day_gen_cc_from_image(in_dir, out_dir)
    #full_season_cc_integrate(cc_dir, out_dir, start_date, end_date)
    #copy_csv_to_outdir(in_dir, out_dir, start_date, end_date)
    '''
    return

if __name__ == "__main__":
    
    main()
    #test()
    
    
