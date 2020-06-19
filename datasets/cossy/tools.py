
def get_video_name(s: str) -> str:
    '''
    Get video name
    '''
    videos = [
        'Meeting1', 'Meeting2', 'Lab1', 'Lab2',
        'Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases', 'High_activity',
        'IRfilter', 'IRill', 'All_off',
        'MW-R',
    ]
    for name in videos:
        if name in s:
            return name
    return 'Unknown video'


def get_video_hw(vname):
    '''
    Get video resolution (height, width)
    '''
    if vname in {'Meeting1', 'Meeting2', 'Lab1', 'Lab2',
                 'Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases'}:
        return (2048, 2048)
    elif vname in {'High_activity', 'IRfilter', 'IRill', 'All_off'}:
        return (1080, 1080)
    else:
        raise NotImplementedError()
