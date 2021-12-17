with open("cdiac_naivetruth_processed.csv", 'r') as f:
    for line in f:
        print(line)
        with open('cdiac_naivetruth_new_processed.csv', 'a') as g:
            new_line = line.replace('/home/cc/CDIACPub8/', '/vol_b/pub8/')
            g.write(new_line)