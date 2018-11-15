"""
author: az
Replace all the *.jpg extensions in the target_folder with *.png extension
"""
import os
from glob import glob

if __name__ == '__main__':
    target_folder = "/home/ale/encoder/kaist"
    filelist = [y for x in os.walk(target_folder) for y in
                glob(os.path.join(x[0], '*.jpg'))]
    for file_obj in filelist:
        """
        try:
            jpg_str = subprocess.check_output(
                    ['file', file_obj], shell=True, stdout=subprocess.PIPE).decode()
            if (re.search("PNG image data", jpg_str, re.IGNORECASE)) or (
                    re.search("Png patch", jpg_str, re.IGNORECASE)):

                old_path = os.path.splitext(file_obj)
                if not os.path.isfile(old_path[0] + '.png'):
                    new_file = old_path[0] + '.png'
                elif not os.path.isfile(file_obj + '.png'):
                    new_file = file_obj + '.png'
                else:
                    print("Found PNG hiding as JPEG but couldn't rename:", file_obj)
                    continue

                print("Found PNG hiding as JPEG, renaming:", file_obj, '->', new_file)
                subprocess.run(['mv', file_obj, new_file])

        except Exception as e:
            logging.error(traceback.format_exc())
        """
        new_name = os.path.splitext(file_obj)[0] + '.png'
        os.rename(src=file_obj, dst=new_name)
    print("Cleaning JPEGs done")
