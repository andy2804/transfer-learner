"""
author: aa
"""
import os

# ====================== USAGE ======================
# Batch renames all files with specified file extension in folder
# Be sure to include all strings occuring in files to be excluded,
# otherwise unwanted files are being renamed
# Always use MAYBE = False to see how files are being handled!
# ===================================================


FROM_STRING = 'ZAURON_'
TO_STRING = 'day_cloudy_'
EXCLUDE_STRING = ['EVENTS']
FILETYPE = '.bag'
FOLDER = '/media/andya/EXTSSD_T3/crossmodal_mt/rosbags/zauron_eye_2.0'
MAYBE = False


def maybe_rename(maybe):
    '''
    Rename all files in folder with specified substring
    :return:
    '''
    files = [f for f in os.listdir(FOLDER) if f.endswith(FILETYPE)]
    for file in files:
        if FROM_STRING in file and not any(string in file for string in EXCLUDE_STRING):
            new_file = file.replace(FROM_STRING, TO_STRING)
            print(file + '-->\n' + new_file + '\n')
            old_path = os.path.join(FOLDER, file)
            new_path = os.path.join(FOLDER, new_file)
            if maybe:
                os.rename(old_path, new_path)


if __name__ == '__main__':
    print('Running batch file rename for dir:\n\t%s\n' % FOLDER)
    print('Looking for files:\t%s' % FILETYPE)
    print('Replacing string:\t%s' % FROM_STRING)
    print('Using string:\t\t%s' % TO_STRING)
    maybe_rename(MAYBE)
