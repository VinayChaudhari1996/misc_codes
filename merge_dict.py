# x = {1:2,3:4,5:6,7:8}
# y = {9:10,11:12,7:13}

# x = {1:'a',2:'b',"c":3,"d":"None"}
# y = {"c":4,"k":9,"d":12}

x = {1:'a',2:['b',1,2,10],"c":{"k","v"}}
y = {"c":4,"k":9,2:12}


def merge_dicts(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].append(dict[key])
            except KeyError:
                d[key] = [dict[key]]
                
    for k,v in d.items():
    
        if len(v)==1:
            d[k] = v[0]
        else:
            pass
                
    return d
    
    
    
    
    
