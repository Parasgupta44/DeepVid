from django.shortcuts import render
from django.conf import settings
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import Video
from .forms import VideoForm
from .model_pred import *
# Create your views here.
def UploadVideo(request):
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        print("############################################################################################")

        tempfilename = ''
        for i in filename:
            if(i!=' '):
                tempfilename +=i
            else:
                tempfilename +='_'
        print(tempfilename)
        uploaded_file_url = fs.url(filename)
        #print(uploaded_file_url)
        form = VideoForm(request.POST , request.FILES)
        #if form.is_valid():
        #    form.save()
        #lastvideo = Video.objects.last()
        print("MEDIA URL MADARCHOD M,ADEGBSTRKGVTRHNGITRGVRT////////////////////////////////////////////////////////////////////////////////")
        #print(settings.MEDIA_URL)
        #print(settings.BASE_DIR)
        #videofile = lastvideo.vid
        #print("LOCATION\n")
        #print(settings.MEDIA_URL + str(videofile))
        #loc = settings.MEDIA_URL + videofile
        vidloc = settings.BASE_DIR + '\media\\' + filename
        print(
            "MEDIA URL MADARCHOD M,ADEGBSTRKGVTRHNGITRGVRT////////////////////////////////////////////////////////////////////////////////")
        print('vidloc\t' + vidloc )
        print(settings.BASE_DIR)
        result = ModelKi(vidloc,settings.BASE_DIR)
        context = {
               'form': form,'media':settings.MEDIA_URL,
                    'vidloc':str('/media/' + filename),
                   'result':result,'uploaded_file_url': uploaded_file_url
               }
        return render(request,'DV/UVresult.html',context)
    else:
        form = VideoForm()

    return render(request,'DV/UV.html',{'form':form})


