import boto3
from datetime import datetime
import json
import logging


class SQS(object):

    def __init__(self, aws_params, sqs_params):
        self.client = None

        self.aws_params = aws_params
        self.sqs_params = sqs_params

        self.interval = self.sqs_params['interval']

    def connected(self):
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = boto3.client(
            'sqs',
            aws_access_key_id=self.aws_params['key'],
            aws_secret_access_key=self.aws_params['secret'],
            region_name=self.aws_params['region']
        )

        return self

    def disconnect(self):
        if self.connected():
            self.client = None

        return self

    def get_queue(self, name):
        self.connect()

        try:
            queue = self.client.get_queue_url(
                QueueName=name
            )['QueueUrl']
        except Exception:
            return

        return queue

    def read_messages(self, queue):
        self.connect()

        messages = []

        mlist = self.client.receive_message(
            QueueUrl=queue,
            MaxNumberOfMessages=self.sqs_params['messages'],
            WaitTimeSeconds=self.sqs_params['wait']
        ).get('Messages', [])

        for m in mlist:
            messages.append(m['Body'])

            self.client.delete_message(
                QueueUrl=queue,
                ReceiptHandle=m['ReceiptHandle']
            )

        return messages

    def send_message(self, queue, message):
        self.connect()

        if isinstance(message, dict):
            message = json.dumps(message)

        self.client.send_message(
            QueueUrl=queue,
            MessageBody=message
        )


class CloudWatchLogHandler(logging.Handler):

    def __init__(self, aws_params, cwl_params):
        super().__init__()

        self.client = None

        self.aws_params = aws_params
        self.cwl_params = cwl_params

        self.fmt = '%(message)s'
        self.dts = datetime.now().strftime('%Y%m%d%H%M%S')

        self.log_group = self.cwl_params['group']
        self.log_streams = {}
        self.sequence_tokens = {}

        self.setFormatter(logging.Formatter(self.fmt))

    def connected(self):
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = boto3.client(
            'logs',
            aws_access_key_id=self.aws_params['key'],
            aws_secret_access_key=self.aws_params['secret'],
            region_name=self.aws_params['region']
        )

        return self

    def disconnect(self):
        if self.connected():
            self.client = None

        return self

    def add_stream(self, name):
        self.connect()

        stream = '{}_{}'.format(self.dts, name)

        self.client.create_log_stream(
            logGroupName=self.log_group,
            logStreamName=stream
        )

        self.log_streams[name] = stream
        self.sequence_tokens[name] = None

        return self

    def emit(self, record):
        self.connect()

        name = record.name
        timestamp = int(record.created * 1000)
        message = self.format(record)

        if name not in self.log_streams:
            self.add_stream(name)

        stream = self.log_streams[name]
        token = self.sequence_tokens[name]

        kwargs = dict(
            logGroupName=self.log_group,
            logStreamName=stream,
            logEvents=[{
                'timestamp': timestamp,
                'message': message
            }]
        )

        if token is not None:
            kwargs['sequenceToken'] = token

        response = None

        try:
            response = self.client.put_log_events(**kwargs)
        except Exception:
            pass
        else:
            if response is not None and 'nextSequenceToken' in response:
                self.sequence_tokens[name] = response['nextSequenceToken']


class Firehose(object):

    def __init__(self, aws_params, kdf_params):
        self.client = None

        self.aws_params = aws_params
        self.kdf_params = kdf_params

    def connected(self):
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = boto3.client(
            'firehose',
            aws_access_key_id=self.aws_params['key'],
            aws_secret_access_key=self.aws_params['secret'],
            region_name=self.aws_params['region']
        )

        return self

    def disconnect(self):
        if self.connected():
            self.client = None

        return self

    def send(self, data):
        self.connect()

        if isinstance(data, dict):
            data = json.dumps(data)

        self.client.put_record(
            DeliveryStreamName=self.kdf_params['stream'],
            Record={
                'Data': data
            }
        )


class Rekognition(object):

    def __init__(self, aws_params, rek_params):
        self.client = None

        self.aws_params = aws_params
        self.rek_params = rek_params

        self.collection = self.rek_params['collection']

    def connected(self):
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = boto3.client(
            'rekognition',
            aws_access_key_id=self.aws_params['key'],
            aws_secret_access_key=self.aws_params['secret'],
            region_name=self.aws_params['region']
        )

        return self

    def disconnect(self):
        if self.connected():
            self.client = None

        return self

    def list(self):
        self.connect()

        result = None

        try:
            response = self.client.list_collections()
            result = response['CollectionIds']
        except Exception:
            pass

        return result

    def create(self, replace=False):
        self.connect()

        result = None

        if replace:
            try:
                self.client.delete_collection(
                    CollectionId=self.collection
                )
            except Exception:
                pass

        try:
            response = self.client.create_collection(
                CollectionId=self.collection
            )
            result = response['CollectionArn']
        except Exception:
            pass

        return result

    def delete(self):
        self.connect()

        result = None

        try:
            response = self.client.delete_collection(
                CollectionId=self.collection
            )
            result = response['StatusCode']
        except Exception:
            pass

        return result

    def describe(self):
        self.connect()

        result = None

        try:
            response = self.client.describe_collection(
                CollectionId=self.collection
            )
        except Exception:
            return result

        result = {
            'id': self.rek_params['collection'],
            'arn': response['CollectionARN'],
            'size': response['FaceCount'],
            'faces': []
        }

        try:
            response = self.client.list_faces(
                CollectionId=self.collection
            )
        except Exception:
            return result

        for face in response['Faces']:
            result['faces'].append({
                'id': face['FaceId'],
                'img_id': face['ExternalImageId']
            })

        return result

    def add(self, image, img_id):
        self.connect()

        result = None

        if isinstance(image, bytes):
            img_obj = {
                'Bytes': image
            }
        elif isinstance(image, str):
            img_obj = {
                'S3Object': {
                    'Bucket': self.rek_params['bucket'],
                    'Name': image
                }
            }
        else:
            return result

        try:
            response = self.client.index_faces(
                CollectionId=self.collection,
                ExternalImageId=img_id,
                Image=img_obj,
                MaxFaces=1
            )
        except Exception:
            return result

        if len(response['FaceRecords']) > 0:
            result = response['FaceRecords'][0]['Face']['ExternalImageId']

        return result

    def search(self, image):
        self.connect()

        result = None

        if isinstance(image, bytes):
            img_obj = {
                'Bytes': image
            }
        elif isinstance(image, str):
            img_obj = {
                'S3Object': {
                    'Bucket': self.rek_params['bucket'],
                    'Name': image
                }
            }
        else:
            return result

        try:
            response = self.client.search_faces_by_image(
                CollectionId=self.collection,
                Image=img_obj,
                MaxFaces=1,
                FaceMatchThreshold=self.rek_params['sim_thr']
            )
        except Exception:
            return result

        if len(response['FaceMatches']) > 0:
            sim = response['FaceMatches'][0]['Similarity']
            if sim >= self.rek_params['sim_thr']:
                result = response['FaceMatches'][0]['Face']['ExternalImageId']

        return result

    def compare(self, source, target):
        self.connect()

        result = None

        if isinstance(source, bytes):
            src_obj = {
                'Bytes': source
            }
        elif isinstance(source, str):
            src_obj = {
                'S3Object': {
                    'Bucket': self.rek_params['bucket'],
                    'Name': source
                }
            }
        else:
            return result

        if isinstance(target, bytes):
            tgt_obj = {
                'Bytes': target
            }
        elif isinstance(target, str):
            tgt_obj = {
                'S3Object': {
                    'Bucket': self.rek_params['bucket'],
                    'Name': target
                }
            }
        else:
            return result

        try:
            response = self.client.compare_faces(
                SourceImage=src_obj,
                TargetImage=tgt_obj,
                SimilarityThreshold=self.rek_params['sim_thr']
            )
        except Exception:
            return result

        for match in response['FaceMatches']:
            if match['Similarity'] >= self.rek_params['sim_thr']:
                return True

        return False


class SNS(object):

    def __init__(self, aws_params, sns_params):
        self.client = None

        self.aws_params = aws_params
        self.sns_params = sns_params

    def connected(self):
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = boto3.client(
            'sns',
            aws_access_key_id=self.aws_params['key'],
            aws_secret_access_key=self.aws_params['secret'],
            region_name=self.aws_params['region']
        )

        return self

    def disconnect(self):
        if self.connected():
            self.client = None

        return self

    def send(self, message, topic=None, subject=None):
        self.connect()

        topic_arn = None

        if topic is not None:
            if topic in self.sns_params['topics'].keys() and self.sns_params['topics'][topic]:
                topic_arn = self.sns_params['topics'][topic]
        else:
            for key, val in self.sns_params['topics'].items():
                if val:
                    topic_arn = self.sns_params['topics'][key]
                    break

        if topic_arn is None:
            return

        if subject is None:
            subject = self.sns_params['subject']

        self.client.publish(
            TopicArn=topic_arn,
            Message=message,
            Subject=subject
        )


class Translate(object):

    def __init__(self, aws_params):
        self.client = None

        self.aws_params = aws_params

    def connected(self):
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = boto3.client(
            'translate',
            aws_access_key_id=self.aws_params['key'],
            aws_secret_access_key=self.aws_params['secret'],
            region_name=self.aws_params['region']
        )

        return self

    def disconnect(self):
        if self.connected():
            self.client = None

        return self

    def translate_text(self, text, source_lang, target_lang):
        self.connect()

        result = None

        try:
            response = self.client.translate_text(
                Text=text,
                SourceLanguageCode=source_lang,
                TargetLanguageCode=target_lang
            )
        except Exception:
            return result

        result = response['TranslatedText']

        return result


class Comprehend(object):

    def __init__(self, aws_params):
        self.client = None

        self.aws_params = aws_params

    def connected(self):
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = boto3.client(
            'comprehend',
            aws_access_key_id=self.aws_params['key'],
            aws_secret_access_key=self.aws_params['secret'],
            region_name=self.aws_params['region']
        )

        return self

    def disconnect(self):
        if self.connected():
            self.client = None

        return self

    def detect_lang(self, text):
        self.connect()

        result = None

        try:
            response = self.client.detect_dominant_language(
                Text=text
            )
        except Exception:
            return result

        if len(response['Languages']) > 0:
            result = response['Languages'][0]['LanguageCode']

        return result

    def detect_entities(self, text, lang=None):
        self.connect()

        if lang is None:
            lang = self.detect_lang(text)

        result = None

        try:
            response = self.client.detect_entities(
                Text=text,
                LanguageCode=lang
            )
        except Exception:
            return result

        result = []
        for entity in response['Entities']:
            result.append({
                'type': entity['Type'].lower(),
                'text': entity['Text'],
                'offset_from': entity['BeginOffset'],
                'offset_to': entity['EndOffset'],
                'score': round(entity['Score'], 4)
            })

        return result

    def detect_sentiment(self, text, lang=None):
        self.connect()

        if lang is None:
            lang = self.detect_lang(text)

        result = None

        try:
            response = self.client.detect_sentiment(
                Text=text,
                LanguageCode=lang
            )
        except Exception:
            return result

        result = {
            'sentiment': response['Sentiment'].lower(),
            'scores': {
                key.lower(): round(val, 4)
                for key, val in response['SentimentScore'].items()
            }
        }

        return result

    def detect_key_phrases(self, text, lang=None):
        self.connect()

        if lang is None:
            lang = self.detect_lang(text)

        result = None

        try:
            response = self.client.detect_key_phrases(
                Text=text,
                LanguageCode=lang
            )
        except Exception:
            return result

        result = []
        for key_phrase in response['KeyPhrases']:
            result.append({
                'text': key_phrase['Text'],
                'offset_from': key_phrase['BeginOffset'],
                'offset_to': key_phrase['EndOffset'],
                'score': round(key_phrase['Score'], 4)
            })

        return result


class Lex(object):

    def __init__(self, aws_params, lex_params, ses_params=None):
        self.client = None

        self.aws_params = aws_params
        self.lex_params = lex_params
        self.ses_params = ses_params

        self.session_id = None

    def connected(self):
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = boto3.client(
            'lexv2-runtime',
            aws_access_key_id=self.aws_params['key'],
            aws_secret_access_key=self.aws_params['secret'],
            region_name=self.aws_params['region']
        )

        session_state = {}
        if 'attributes' in self.ses_params:
            session_state['sessionAttributes'] = self.ses_params['attributes']

        response = self.client.put_session(
            botId=self.lex_params['bot_id'],
            botAliasId=self.lex_params['bot_alias_id'],
            localeId=self.lex_params['locale_id'],
            sessionId=self.ses_params['session_id'],
            sessionState=session_state
        )
        self.session_id = response['sessionId']

        return self

    def disconnect(self):
        if self.connected():
            self.client.delete_session(
                botId=self.lex_params['bot_id'],
                botAliasId=self.lex_params['bot_alias_id'],
                localeId=self.lex_params['locale_id'],
                sessionId=self.ses_params['session_id']
            )
            self.session_id = None
            self.client = None

        return self

    def send(self, text):
        self.connect()

        result = None

        try:
            response = self.client.recognize_text(
                botId=self.lex_params['bot_id'],
                botAliasId=self.lex_params['bot_alias_id'],
                localeId=self.lex_params['locale_id'],
                sessionId=self.session_id,
                text=text
            )
            if len(response['messages']) > 0:
                result = response['messages'][0]['content']
        except Exception:
            pass

        return result
