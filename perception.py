 def get_my_data():
    my_data_x=[]
    my_data_x.append([1,0,0]) #treshold daki a[0] değeri hep 1 olacak
    my_data_x.append([1,0,1])
    my_data_x.append([1,1,0])
    my_data_x.append([1,1,1])
    my_data_x
    my_data_y=[]
    my_data_y.append(0)
    my_data_y.append(0)
    my_data_y.append(0)
    my_data_y.append(1)
    my_data_y
   
    return my_data_x,my_data_y
# In[50]:
x,y=get_my_data()
for a,b in zip(x,y):
    print(a,b)
   
#4 tane sample var, 3 tanesi a dan 1 tanesi b den
# In[53]:
def get_parameters():
    w=[]
    w.append(30)  #random olarak oluşturuldu
    w.append(20)
    w.append(10)
    w
    learning_rate=1 #random olarak oluşturuldu
    epoch=100
   
    return w,learning_rate,epoch
get_parameters()
# In[54]:
w,learning_rate,epoch=get_parameters()
samples,output=get_my_data() # sample -> x değerleri
samples
# In[59]:
for i in range(1000):
    error="hata_yok"
    s=-1
    print("************************************************")
    for each_sample,d in zip(samples,output): #
        s=s+1
        print("ağırlık :",w)
        print("örnek ",each_sample)
        print("Gerçek output ",d)
       #print(my_product_two_dim_with_threshold(w,each_sample))
       
        u=my_product_two_dim_with_threshold(each_sample,w)
        #print(u)
        if(u>0):
            y=1
        else:
            y=0
       
        print("tahmin çıktı:",y)
        print("")
       
       
        if(y!=d): #error var #d gerçek değer #y bizim bulduğumuz değer
            for s in range(3):
                w[s]=w[s]-learning_rate*(y-d)*each_sample[s]
            error="hata_var"
    if(error=="hata_yok"):
        print("hata yok", i)
        break #return 0
print(w)

xor


def my_product_two_dim_with_threshold(a,b): # u->w.x skaler çarpım

    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def get_my_data():
    my_data_x=[]
    #XOR İÇİN EĞİTİLEMİYOR
    #PERCEPTRON UZAYDA BİR DOĞRU İLE AYRILABİLEN PROBLEMLERİ ÇÖZEBİLİYOR O YÜZDEN XOR ÇÖZÜLEMEZ
    my_data_x.append([1,0,0]) #treshold daki a[0] değeri hep 1 olacak
    my_data_x.append([1,0,1])
    my_data_x.append([1,1,0])
    my_data_x.append([1,1,1])
    my_data_x
    my_data_y=[]
    my_data_y.append(0)
    my_data_y.append(1)
    my_data_y.append(1)
    my_data_y.append(0)
    my_data_y
   
    return my_data_x,my_data_y
# In[70]:
x,y=get_my_data()
for a,b in zip(x,y):
    print(a,b)
   
#4 tane sample var, 3 tanesi a dan 1 tanesi b den
# In[71]:
def get_parameters():
    w=[]
    w.append(30)  #random olarak oluşturuldu
    w.append(20)
    w.append(10)
    w
    learning_rate=1 #random olarak oluşturuldu
    epoch=100
   
    return w,learning_rate,epoch
get_parameters()
# In[72]:
w,learning_rate,epoch=get_parameters()
samples,output=get_my_data() # sample -> x değerleri
samples
# In[68]:
for i in range(100):
    error="hata_yok"
    s=-1
    print("************************************************")
    for each_sample,d in zip(samples,output): #
        s=s+1
        print("ağırlık :",w)
        print("örnek ",each_sample)
        print("Gerçek output ",d)
       #print(my_product_two_dim_with_threshold(w,each_sample))
       
        u=my_product_two_dim_with_threshold(each_sample,w)
        #print(u)
        if(u>0):
            y=1
        else:
            y=0
       
        print("tahmin çıktı:",y)
        print("")
       
       
        if(y!=d): #error var #d gerçek değer #y bizim bulduğumuz değer
            for s in range(3):
                w[s]=w[s]-learning_rate*(y-d)*each_sample[s]
            error="hata_var"
    if(error=="hata_yok"):
        print("hata yok", i)
        break #return 0
print(w)
