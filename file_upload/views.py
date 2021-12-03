from django.shortcuts import render
from .classifier import PDFReader
from django.core.files.storage import FileSystemStorage


# Create your views here.
def index(request):
    return render(request, 'file_upload/index.html')


def file_get(request):
    print("______________________")
    try:
        if (request.method == 'POST') and request.FILES['upload']:
            form = request.FILES['upload']
            fss = FileSystemStorage()
            file = fss.save(form.name, form)
            file_url = fss.url(file)
            print(form)
            pdf_reader = PDFReader()
            df = pdf_reader.CheckTone(file_url)
            fss.delete(file)
            return render(request, 'file_upload/results.html', {'file_name': form, 'data': df})
    except Exception as e:
        print("Error: ", e)
        return render(request, 'file_upload/results.html')
