import os
import json
import time
import numpy as np
import multiprocessing
from scipy.sparse import csr_matrix, vstack
from Preprocessing import Driver


start = time.time()


def get_io_addr_day_samp():
    # may = [(5, i) for i in range(1, 8)]
    may = []
    june = [(6, i) for i in range(4, 6)]
    # june = []

    root = "/home/wlu/Desktop/rips16"
    filename_in = "day_samp_raw"
    filename_out = "day_samp_new_bin.npy"

    list_io_addr = []
    for item in may+june:
        month = item[0]
        day = item[1]
        io_addr = os.path.join(root,
                               str(month).rjust(2, "0"),
                               str(day).rjust(2, "0"))
        addr_in = os.path.join(io_addr, filename_in)
        addr_out = os.path.join(io_addr, filename_out)
        list_io_addr.append((addr_in, addr_out))

    return list_io_addr


def get_io_addr_random_sample():
    list_io_addr = []
    root = "/home/ubuntu/random_samples"
    prefix = ["all"]
    suffix = [i for i in range(6)]
    for i in prefix:
        for j in suffix:
            file_name = i+"data"+str(j)
            addr_in = os.path.join(root, file_name+".txt")
            addr_out = os.path.join(root, file_name+"_new.npy")
            list_io_addr.append((addr_in, addr_out))
    return list_io_addr


def crawl(io_addr):
    dumped = 0
    data_sparse_list = []
    # for suffix in ["pos"]:
    addr_in = io_addr[0]
    addr_out = io_addr[1]
    # addr_in = addr_in + suffix
    # addr_out = addr_out + suffix + ".npy"
    if os.path.isfile(addr_in):
        with open(addr_in, "r") as file_in:
            print "Processing {}".format(addr_in)
            for line in file_in:
                # try:
                entry = json.loads(line)
                result = []
                Driver.process(entry, result)
                data_sparse_list.append(csr_matrix(result))

                # except:
                #     dumped += 1

        data_matrix = vstack(data_sparse_list)
        with open(addr_out, 'w') as file_out:
            np.savez(file_out,
                     data=data_matrix.data,
                     indices=data_matrix.indices,
                     indptr=data_matrix.indptr,
                     shape=data_matrix.shape)

    else:
        print "\nFile Missing: {}\n".format(addr_in)

    return dumped


if __name__ == '__main__':
    cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpus)
    list_io_addr = get_io_addr_day_samp()

    dumped = 0
    for result in p.imap(crawl, list_io_addr):
        dumped += result

    print "{} lines dumped".format(dumped)

print "Completed in {} seconds\n".format(round(time.time()-start, 2))
