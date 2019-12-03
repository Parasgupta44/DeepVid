from django.db import models

# Create your models here.

from django.db import models



class Video(models.Model):

    vid = models.FileField()
    def __str__(self):
        print()