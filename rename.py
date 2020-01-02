import os 

def rename(dirs):
    i = 0
        
    for filename in os.listdir(dirs): 

        dst = str(i) + ".jpg"
        
        src = os.path.join(dirs, filename)
        dst = os.path.join(dirs, dst)
        
        os.rename(src, dst) 
        
        i += 1
    
    print(str(i)+ " images in " + dirs)

rename("blessed")
print("Blessed renamed")

rename("cursed")
print("Cursed renamed")