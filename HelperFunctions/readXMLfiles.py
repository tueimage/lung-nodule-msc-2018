# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 09:28:35 2018

@author: s120116
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:34:49 2018

@author: s120116


"""



import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas

import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3

import math

from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob

def find_mhd_file(patient_id):
    for subject_no in range(0,10 ):
     
        src_dir = 'C:/Users/s120116/Documents/LUNAsubsets/' + "subset" + str(subject_no) + "/"
     
        for src_path in glob.glob(src_dir + "*.mhd"):
            if patient_id in src_path:
                return src_path
    return None



def load_lidc_xml(xml_path, agreement_threshold=0, only_patient=None, save_nodules=False):
    
    pos_lines = []
    neg_lines = []
    extended_lines = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")

    if xml.LidcReadMessage is None:
        return None, None, None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

    if only_patient is not None:
        if only_patient != patient_id:
            return None, None, None

    src_path = find_mhd_file(patient_id)
    print(src_path)
    if src_path is None:
       return None, None, None

    print(patient_id)
    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
  #  rescale = spacing / settings.TARGET_VOXEL_MM

    reading_sessions = xml.LidcReadMessage.find_all("readingSession")
    for reading_session in reading_sessions:
        # print("Sesion")
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            # print("  ", nodule.noduleID)
            rois = nodule.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue

            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue

            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
          #  z_center -= origin[2]
           # z_center /= spacing[2]

            x_center_perc = round(x_center* spacing[0] + origin[0], 4)
            y_center_perc = round(y_center * spacing[1] + origin[1], 4)
            #z_center_perc = round(z_center / img_array.shape[0], 4)
      
            diameter= round(max(x_diameter , y_diameter, 4))
       
          #  diameter_perc = round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)

            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue

            malignacy = nodule.characteristics.malignancy.text
            sphericiy = nodule.characteristics.sphericity.text
            margin = nodule.characteristics.margin.text
            spiculation = nodule.characteristics.spiculation.text
            texture = nodule.characteristics.texture.text
            calcification = nodule.characteristics.calcification.text
            internal_structure = nodule.characteristics.internalStructure.text
            lobulation = nodule.characteristics.lobulation.text
            subtlety = nodule.characteristics.subtlety.text

            line = [nodule_id, x_center_perc, y_center_perc, z_center, diameter, malignacy]
            extended_line = [patient_id, nodule_id, x_center_perc, y_center_perc, z_center, diameter, malignacy, origin, spacing, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
            pos_lines.append(line)
            extended_lines.append(extended_line)

        nonNodules = reading_session.find_all("nonNodule")
        for nonNodule in nonNodules:
            z_center = float(nonNodule.imageZposition.text)
            #z_center -= origin[2]
            #z_center /= spacing[2]
            x_center = int(nonNodule.locus.xCoord.text)
            y_center = int(nonNodule.locus.yCoord.text)
            nodule_id = nonNodule.nonNoduleID.text
            x_center_perc = round(x_center)# / img_array.shape[2], 4)
            y_center_perc = round(y_center )#/ img_array.shape[1], 4)
            z_center_perc = round(z_center )#)/ img_array.shape[0], 4)
            diameter_perc = 3 #round(max(6 / img_array.shape[2], 6 / img_array.shape[1]), 4)
            # print("Non nodule!", z_center)
            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0]
            neg_lines.append(line)

    if agreement_threshold > 1:
 
        filtered_lines = []
        
        for pos_line1 in pos_lines:
            id1 = pos_line1[0]
            
            x1 = pos_line1[1]
            y1 = pos_line1[2]
            z1 = pos_line1[3]
            d1 = pos_line1[4]
            m1 = pos_line1[5]
            overlaps = 0
            for pos_line2 in pos_lines:
                id2 = pos_line2[0]
                if id1 == id2:
                    continue
                
                x2 = pos_line2[1]
                y2 = pos_line2[2]
                z2 = pos_line2[3]
                d2 = pos_line1[4]
                m2 =  pos_line2[5]
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                
                if dist < d1 or dist < d2:
                    print(overlaps)
                    overlaps += 1
                    pos_line1.append(m2)
            if overlaps >= agreement_threshold:
                print(pos_line1)
                filtered_lines.append(pos_line1)
            # else:
            #     print("Too few overlaps")
        pos_lines = filtered_lines

    #df_annos = pandas.DataFrame(pos_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"  ])
    #df_annos.to_csv( "C:/Users/s120116/Documents/labels/" + patient_id + "_annos_pos_lidc.csv", index=False)
    #df_neg_annos = pandas.DataFrame(neg_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    #df_neg_annos.to_csv( "C:/Users/s120116/Documents/labels/" + patient_id + "_annos_neg_lidc.csv", index=False)

    # return [patient_id, spacing[0], spacing[1], spacing[2]]
    return pos_lines, neg_lines, extended_lines


def process_lidc_annotations(only_patient=None, agreement_threshold=3):
    # lines.append(",".join())
    file_no = 0
    pos_count = 0
    neg_count = 0
    all_lines = []

    for anno_dir in [d for d in glob.glob("C:/tcia-lidc-xml/*") if os.path.isdir(d)]:
        xml_paths = glob.glob(anno_dir + "/*.xml")
        for xml_path in xml_paths:
            print(file_no, ": ",  xml_path)
            pos, neg, extended = load_lidc_xml(xml_path=xml_path, only_patient=None, agreement_threshold=3)
           
            if pos is not None:
                pos_count += len(pos)
                neg_count += len(neg)
                print("Pos: ", pos_count, " Neg: ", neg_count)
                file_no += 1
                all_lines += pos

                
            # if file_no > 10:
            #     break

            # extended_line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
    df_annos = pandas.DataFrame(all_lines, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore", "origin",  "spacing", "sphericiy", "margin", "spiculation", "texture", "calcification", "internal_structure", "lobulation", "subtlety"])
    #df_annos = pandas.DataFrame(pos_lines, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore0","malscore1", 'malscore2', 'malscore3' ])
    df_annos.to_csv("lidc_annotations_nodulesWithAllMalignancies.csv", index=False)
    return pos_lines
    
pos_lines=process_lidc_annotations(agreement_threshold=3)    
    