import shutil
import xlrd
import re
import os

img_dir = 'image_x/'
res_dir = 'result/'

res_file = 'result.xlsx'

res_xlsx = xlrd.open_workbook('result.xlsx')
booksheet = res_xlsx.sheet_by_index(0)

for row in range(1, booksheet.nrows):
    img_name = str(booksheet.cell(row, 1).value)
    dr_level = int(booksheet.cell(row, 2).value)
    dme_level = int(booksheet.cell(row, 3).value)
    referable = int(booksheet.cell(row, 4).value)
    if dr_level > 0:
        if os.path.exists(img_dir + img_name) == True:
            shutil.copy(img_dir+img_name, res_dir + str(dr_level)+'/')
            print(img_name, dr_level)
