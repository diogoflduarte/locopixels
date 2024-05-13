import os

#import CareyFileProcessing wtf?
import CareyUtils
import numpy as np
import shutil
import datetime

def standardizeBonsaiVideoFileNames(input_dir, extensions=['.avi', '.csv'],
                                    mouse_name='MC0000', session='S1',
                                    delete=False,
                                    log=True,
                                    dryrun=False):
    # reads the file list in input_dir and changes the naming to the carey lab convention

    # iterate through the extension list and separate the file name into its components
    # starts with the avi filename and expects a CSV file that matches the end name of the structure

    # 1 - get file name of avi file
    # 2 - store corresponding date and time
    # 3 - confirm that there's a matching CSV file
    # 4 - segment file into mouse name

    ct_tol = datetime.timedelta(0,3)

    file_list = list()
    num_extensions = len(extensions)
    for ii in range(num_extensions):
        file_list.append(list())

    # organize files into lists by extension
    all_files = os.listdir(input_dir)
    for file in all_files:
        for ext in range(num_extensions):
            if file.endswith(extensions[ext]):
                file_list[ext].append(file)

    # iterate through list 1, which is avi, and sort the files by
    # creation date. for each file, find a matching CSV file

    creation_times = list()
    for file_idx in range(len(file_list[0])):
        ct = CareyUtils.getBonsaiDateTime(file_list[0][file_idx])
        creation_times.append(ct)

    order = np.argsort(creation_times)

    if file_list is not None and log is True:
        # create log file
        flog = open(os.path.join(input_dir, 'log.txt'), 'a')


    # for all indices in order, set the trial number
    for trial in order:
        in_file = file_list[0][trial]
        matching_files = list()

        # find matching other extension (csv) file
        ct_thisfile = CareyUtils.getBonsaiDateTime(in_file)
        for ext in range(1,len(extensions)):
            for file in file_list[ext]:
                ct_candidate = CareyUtils.getBonsaiDateTime(file)
                if abs(ct_thisfile-ct_candidate) <= ct_tol:
                    matching_files.append(file)


        fileparts = in_file.split('_')

        if mouse_name is None:
            # find mouse name from filename
            mouse_name = fileparts[0] + '_' + fileparts[1]

        if session is None:
            # find session from folder (input_dir)
            session = os.path.split(input_dir)[1]

        outfile = mouse_name + '_' + fileparts[2][:-3] + '_' + session + '_' + str(trial+1)

        if dryrun:
            print('DRY RUN    ' + in_file + ' renamed ' + outfile + os.path.splitext(in_file)[1])
        else:
            if delete:
                os.rename(os.path.join(input_dir, in_file),
                          os.path.join(input_dir, outfile + os.path.splitext(in_file)[1]))
            else:
                shutil.copyfile(os.path.join(input_dir, in_file),
                            os.path.join(input_dir, outfile + os.path.splitext(in_file)[1]))

            print(in_file + ' renamed ' + outfile + os.path.splitext(in_file)[1])
            if log:
                flog.write(in_file + ' renamed ' + outfile + os.path.splitext(in_file)[1])
                flog.write('\n')

            for jj in matching_files:
                fileparts_match = jj.split('_')
                outfile_match = mouse_name + '_' + fileparts_match[2][:-3] + '_' + session + '_' + str(trial+1)
                if delete:
                    os.rename(os.path.join(input_dir, jj),
                              os.path.join(input_dir, outfile_match + os.path.splitext(jj)[1]))
                else:
                    shutil.copyfile(os.path.join(input_dir, jj),
                                os.path.join(input_dir, outfile_match + os.path.splitext(jj)[1]))
                print(jj + ' renamed ' + outfile + os.path.splitext(jj)[1])
                if log:
                    flog.write(jj + ' renamed ' + outfile + os.path.splitext(jj)[1])

    if log:
        flog.close()

def getFileProps(filename, standard='HGM', name_contains_underscore=True, stripext=True):
    """
    gets the properties from the filename
    :param filename:
    :param standard:
    :param name_contains_underscore:
    :return: file_props, file_parts, extension
    """
    __, filename = os.path.split(filename)
    __, extension = os.path.splitext(filename)

    if stripext:
        filename = filename[:-len(extension)]

    if standard == 'HGM':
        file_props = {}
        file_parts = filename.split('_')
        file_props['name']           = file_parts[0]
        file_props['weight']         = file_parts[1]
        file_props['age']            = file_parts[2]
        file_props['sex']            = file_parts[3]
        file_props['condition']      = file_parts[4]
        file_props['belt_speed_L']   = file_parts[5]
        file_props['belt_speed_R']   = file_parts[6]
        file_props['session']        = file_parts[7]
        file_props['trial']          = file_parts[8]
        file_props['suffix']         = file_parts[9]

    if standard == 'NPXRIG':
        file_props = {}
        file_parts = filename.split('_')
        file_props['trial']             = int(file_parts[-1]) # pay attention to this, name might be split in underscore
        file_props['session']           = file_parts[-2]
        if name_contains_underscore:
            file_props['mouse']         = '_'.join(file_parts[0:2])
        else:
            file_props['mouse']         = file_parts[0]

    if standard == 'NPXRIG_DLC':
        file_props = {}
        file_parts = filename.split('_')
        if name_contains_underscore:
            file_props['mouse'] = '_'.join(file_parts[0:2])
        else:
            file_props['mouse'] = file_parts[0]

        file_props['session']   = file_parts[3]
        if file_parts[4].__contains__('DLC'):
            file_props['trial']     = int(file_parts[4][0:-3])
        else:
            file_props['trial']     = int(file_parts[4])

    if standard == 'BONSAI_TO_DLC':
        file_props = {}
        file_parts = filename.split('_')
        if name_contains_underscore:
            file_props['mouse'] = '_'.join(file_parts[0:2])
        else:
            file_props['mouse'] = file_parts[0]

        file_props['session']   = file_parts[3]
        file_props['trial']     = int(file_parts[4])


    return file_props, file_parts, extension

def sortFilesByTrial(file_list, standard='NPXRIG_DLC', name_contains_underscore=True, stripext=True):
    """
    sortFilesByTrial gets the file properties and returns the sorted list,
    indices and original file list
    :param file_list:
    :param standard:
    :return: sorted_list, ordered_indices, file_list
    """

    trial_list = []
    for idx, file in enumerate(file_list):
        file_props, __, __ = getFileProps(file, standard=standard,
                                            name_contains_underscore=name_contains_underscore,
                                            stripext=stripext)
        trial_list.append(file_props['trial'])

    trials = np.asarray(trial_list)
    ordered_indices = np.argsort(trials)
    sorted_list = [file_list[i] for i in ordered_indices]

    return sorted_list, ordered_indices, file_list
def buildFilenameFromProps(file_props, extension, standard='HGM'):

    if standard == 'HGM':
        if isinstance(file_props, list):
            filename = '_'.join(file_props) + extension
        elif isinstance(file_props, dict):
            file_props = list(file_props.values())
            # a bit of semi-elegant recursion
            filename = buildFilenameFromProps(file_props, extension)

    return filename

class FileOps():
    def __init__(self, target):
        self.target = target

    def listdir(self):
        self.file_list = os.listdir(self.target)
        return self

    def filterFilesFromList(self, pattern):
        self.file_list = CareyUtils.filterFilesFromList(self.file_list, pattern)
        return self
