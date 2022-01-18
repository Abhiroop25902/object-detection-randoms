FILENAME = './coco_labels.txt'

class CocoLabelName:
    def __init__(self):
        self._coco_dict = {}
        for line in open(FILENAME, 'r').readlines():
            line = line.rstrip('\n')
            phrases = line.split(',')
            key = int(phrases[0].split(':')[1])
            value = phrases[1].split(':')[1]
            self._coco_dict[key] = value


    def getName(self, label: int) -> str:
        return self._coco_dict[label]