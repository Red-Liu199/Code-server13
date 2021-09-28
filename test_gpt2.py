def a(name, **kwargs):
    if 'goal' in kwargs:
        print('goal here',kwargs['goal'])
a('test', goal='test')