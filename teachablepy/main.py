from teachablepy.teachable_hotword import TeachableHotwordDetection

s = TeachableHotwordDetection("teachablepy\model\soundclassifier.tflite", probability_threshold=0.5)
s.start()