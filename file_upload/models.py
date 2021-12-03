from django.db import models


class Sentimental(models.Model):
    doc_name = models.FileField(blank=True)
    result = models.CharField(blank=True, max_length=5000)

    def __str__(self):
        return str(self.doc_name)
