from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',views.hoempage,name='Welcome'),
    path('homepage.html',views.hoempage,name='Welcome'),
    path('content.html',views.content,name='content'),
    path('history.html',views.history,name='history'),
    path('mic.html',views.mic,name='Mic'),
    path('file.html',views.file,name='File'),
    path('predictEmotion',views.predictEmotion,name='result'),
    path('mic.html',views.record,name='record'),
]


urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
