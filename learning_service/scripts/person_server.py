#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from learning_service.srv import Person, PersonResponse

def personCallback(req):
	rospy.loginfo("Person: name:%s, age:%d, sex:%d", req.name, req.age, req.sex)

	return PersonResponse("OK")

def person_server():
	rospy.init_node('person_server')

	s = rospy.Service('/show_person', Person, personCallback)

	print("Ready to show person infomation.")
	rospy.spin()

if __name__ == '__main__':
	person_server()