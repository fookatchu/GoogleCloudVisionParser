from errbot import BotPlugin, re_botcmd, botcmd
import re
import requests
import base64
import io
import time
import datetime as dt
from collections import OrderedDict
# from PIL import Image, ImageDraw
from PIL import Image, ImageDraw
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import traceback
try:
    from IPython import embed
except:
    pass

DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
MAX_DOWNLOAD_SIZE = 5000000
MAX_IMAGE_SIZE = 1600
MAX_RESULTS = 10
DAILY_QUOTA = 30

likelihood_format = {
    'UNKNOWN': '[__?__]',
    'VERY_UNLIKELY': '[+____]',
    'UNLIKELY': '[++___]',
    'POSSIBLE': '[+++__]',
    'LIKELY': '[++++_]',
    'VERY_LIKELY': '[+++++]',
}


def resize(image):
    width, height = image.size
    # print(width, height)
    if max(image.size) < MAX_IMAGE_SIZE:  # shortpath
        return image
    if width > height:
        new_width = MAX_IMAGE_SIZE
        scale_factor = new_width / float(width)
        new_height = int((float(height) * scale_factor))
    else:
        new_height = MAX_IMAGE_SIZE
        scale_factor = new_height / float(height)
        new_width = int((float(width) * scale_factor))
    image = image.resize((new_width , new_height), Image.ANTIALIAS)
    print(image.size)
    return image


def format_safe_search(annotations):
    od = OrderedDict()
    for key, val in sorted(annotations.items()):
        od[key] = likelihood_format.get(val, likelihood_format['UNKNOWN'])
    return od


def format_face(annotations):
    od = OrderedDict()
    for key, val in sorted(annotations.items()):
        if 'Likelihood' in key:
            od[key.replace('Likelihood', '')] = likelihood_format.get(val, likelihood_format['UNKNOWN'])
    return od


def get_vision_service():
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials,
                              discoveryServiceUrl=DISCOVERY_URL)
    import os
    print(credentials.serialization_data)   
    print(os.getcwd())
    print(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    return service


def get_image(url):
    r = requests.get(url, verify=False, stream=True, timeout=2)
    if 'image' not in r.headers['content-type']:
        raise UserWarning('not an image file!')
    if int(r.headers['Content-Length'])  > MAX_DOWNLOAD_SIZE:
        raise UserWarning('Image file too big!')
    r.raw.decode_content = True
    image = io.BytesIO(r.raw.read())
    image = Image.open(image)
    resized_image = resize(image)
    out_file = io.BytesIO()
    resized_image.save(out_file, format='jpeg')
    out_file.seek(0)
    return base64.b64encode(out_file.read())


class GoogleCloudVisionParser(BotPlugin):
    """Parses links and extracts embedded information"""

    def check_quota(self):
        today = dt.date.today()
        try:
            calls, date = self['QUOTA']
            if (date - today) >= dt.timedelta(1):
                self['QUOTA'] = (0, today)
        except KeyError:
            self['QUOTA'] = (0, today)
        return self['QUOTA']

    def increment_quota(self):
        calls, date = self['QUOTA']
        calls = calls + 1
        self['QUOTA'] = (calls, date)

    @botcmd
    def quota(self, msg, args):
        calls, date = self.check_quota()
        yield 'current quota is {} of {} for today.'.format(calls, DAILY_QUOTA)

    @botcmd(admin_only=True)
    def reset_quota(self, msg, args):
        yield 'resetting quota'
        self['QUOTA'] = (0, dt.date.today())

    @re_botcmd(prefixed=False, flags=re.IGNORECASE, pattern='(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]|\.(jpe?g|png|gif|bmp)))+)')
    def img_url_match(self, msg, match):
        url = match.groups()[0]
        self.log.info('got url: {}'.format(url))
        try:
            image_content = get_image(url)
        except UserWarning:
            self.log.exception(traceback.format_exc())
            return
        except requests.exceptions.ConnectionError:
            self.log.exception(traceback.format_exc())
            return
        except:
            self.log.exception(traceback.format_exc())
            yield 'crunshing image...'
            time.sleep(2)
            yield 'not! :-p'
            return
        
        calls, date = self.check_quota()
        if calls >= DAILY_QUOTA:
            yield 'max calls per day reached!'
            return

        yield 'crunshing image...'
        # storing the credentials didn't work (?!), so they will be build everytime.
        # see https://cloud.google.com/vision/docs/getting-started on how to get an api key
        credentials = GoogleCredentials.get_application_default()
        service = discovery.build('vision', 'v1', credentials=credentials,
                              discoveryServiceUrl=DISCOVERY_URL)
        service_request = service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': image_content.decode('UTF-8')
                },
                'features': [
                {
                    'type': 'LABEL_DETECTION',
                    'maxResults': MAX_RESULTS
                },
                {
                    'type': 'FACE_DETECTION',
                    'maxResults': MAX_RESULTS
                },
                {
                    'type': 'SAFE_SEARCH_DETECTION'
                }
                ]
            }]
        })
        response = service_request.execute()
        try:
            faces = response['responses'][0]['faceAnnotations']
            yield '======= FACES ======='
            for i, face in enumerate(faces):
                yield 'FACE #{}:'.format(i)
                face_results = format_face(face)
                for key, val in face_results.items():
                    yield '{:>15}: {}'.format(val, key)
                yield ''
        except KeyError:  # no faces detected
            pass

        safe_search_results = format_safe_search(response['responses'][0]['safeSearchAnnotation'])
        yield '======= SFW ======='
        for key, val in safe_search_results.items():
            yield '{:>15}: {}'.format(val, key)
        yield ''
        yield '======= TAGS ======='
        yield ', '.join([item['description'] for item in response['responses'][0]['labelAnnotations']])
        self.increment_quota()
        
