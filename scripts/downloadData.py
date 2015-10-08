
import urllib
import os


def download_data(dataset):
    '''
    Download 20Newsgroups.mat if this file doesn't already exist.
    Source: www.cad.zju.edu.cn/home/dengcai/Data/20Newsgroups/20Newsgroups.mat
    :param dataset: dataset/20Newsgroups.mat
    :return:
    '''

    ###################
    #   Download Data #
    ###################

    # Download the 20 news dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == '20Newsgroups.mat':
        origin = ('http://www.cad.zju.edu.cn/home/dengcai/Data/20Newsgroups/20Newsgroups.mat')
        print('Downloading data from %s' % origin)
        urllib.urlretrieve(origin, dataset)

if __name__ == '__main__':
    download_data('dataset/20Newsgroups.mat')
