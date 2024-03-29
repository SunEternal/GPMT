#generating practical pcap files
import dpkt
import random
import shutil
import traceback
import os
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print (path+' create successfully')
        return True
    else:
        return False

# g = G_model["MLP"]

for model_name in G_model:
#     model_name = "FS"
    g = G_model[model_name]
    tmp_list = []
    print(model_name,"is generate")
#     shutil.rmtree("./fake_data/"+model_name)  
    mkdir("./fake_data/"+model_name)
    for i in range(len(train_name2)):
        if random.random()>0.05:
            continue
        else:
            pass
        file_pcap_name = str(train_name2[i][0])
        f=open(file_pcap_name,"rb")
        whole_file=f.read()
        outFile = bytearray(whole_file)
        tmp=32
        j = 0
#         try:
        pre_time1=struct.unpack('<I',whole_file[tmp-8:tmp-4])[0]
        pre_time2=struct.unpack('<I',whole_file[tmp-4:tmp])[0]
        all_time1 = 0
        all_time2 = 0
        en = g(torch.tensor(train_adv2[i:i+1]).to(device))[0].clamp(0,1).cpu()
        en = decode_encode(en,x_min,x_max)[0] 
        lengths = en[0::2].clamp(0,1200)[1:]
        en = (en[1::2]).clamp(0,1)
        
#         en = torch.rand([127])
        
        try:
            while outFile[tmp:tmp+4]:
                tmp_list.append(tmp)
            #     length = len(packet[1])
                length2 = struct.unpack('<I',outFile[tmp+4:tmp+8])[0]
                tmp += length2+16
                if j>=len(en):
                    pre_time1=struct.unpack('<I',outFile[tmp-8:tmp-4])[0]
                    pre_time2=struct.unpack('<I',outFile[tmp-4:tmp])[0]
                    #Keep the timestamp of subsequent packets delayed
                    outFile[tmp-8:tmp-4] = struct.pack('<I',((all_time1+pre_time2)//1000000)+pre_time1)
                    outFile[tmp-4:tmp] = struct.pack('<I',(all_time2+pre_time2)%1000000)
                    continue
                elif outFile[tmp:tmp+4]:
                    time1 = int(en[j])
                    time2 = int(en[j]*1000000%1000000)
                    all_time1 += time1 + int(en[j]*1000000//1000000)
                    all_time2 += int(en[j]*1000000)
                    all_time2 = all_time2%1000000
#                     pre_time1=struct.unpack('<I',whole_file[tmp-8:tmp-4])[0]
#                     pre_time2=struct.unpack('<I',whole_file[tmp-4:tmp])[0]
#                     if time2>1000000:
                    pre_time1=((time2+pre_time2)//1000000)+pre_time1+time1
                    pre_time2 = (time2+pre_time2)%1000000
                        
                    #add time perturbations
                    outFile[tmp-8:tmp-4] = struct.pack('<I',pre_time1)
                    outFile[tmp-4:tmp] = struct.pack('<I',pre_time2)
                    
                    length2 = struct.unpack('<I',outFile[tmp+4:tmp+8])[0]
                    
#                     add length perturbations
                    length = int(lengths[j]) - length2
                    if length > 0:
                        cacheFile = outFile[:tmp+length2+8]
                        cacheFile.extend([0]*(length))
                        outFile = cacheFile+outFile[tmp+length2+8:]
                        outFile[tmp+4:tmp+8] = struct.pack('<I',length+length2)
                        outFile[tmp:tmp+4] = struct.pack('<I',struct.unpack('<I',outFile[tmp:tmp+4])[0]+length)

#                         print("yes")
                else:
        #                 print("tmp is",tmp)
                    break
                j+=1
        except Exception as e:
            traceback.print_exc()
#             print(j)
#             pass
    #     print("time1",time1)
        w = open("./fake_data/"+model_name+"/"+file_pcap_name.split("/")[-1],"wb")
        w.write(outFile)
        w.close()
#     break


# merge and generate pcap file
# os.popen("cd ./15Black/")
tmp = os.popen('ls '+path).readlines()
tmp = [i[:-1] for i in tmp]

for i in range(len(tmp)//100):
    sub_fake_pcap = path.join(tmp[i*100:(i+1)*100])
    os.popen('mergecap -w ./all_fake_pcap/'+str(i)+path+sub_fake_pcap).readlines()
