#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed May 27 21:40:40 2020

@author: AtulHome
"""

import cv2 as cv
from google.colab.patches import cv2_imshow
from PIL import Image
from scipy import ndimage
import copy
from PIL import Image as PILImage
import numpy as np
import math

import cv2 as cv
from google.colab.patches import cv2_imshow
from PIL import Image
from scipy import ndimage
import copy
from PIL import Image as PILImage
import numpy as np
import math


class car(object):

  # x and y are center points of the car

    def __init__(
        self,
        x,
        y,
        angle,
        ):
        self.x = x
        self.y = y
        (self.length, self.width) = (int(20), int(10))
        self.angle = angle

    def move(
        self,
        velocity_x,
        velocity_y,
        rotation,
        ):
        self.x = self.x + velocity_x
        self.y = self.y + velocity_y
        self.angle = self.angle + rotation

        if self.angle > 360:
            self.angle = self.angle % 360
        elif self.angle < -360:
            self.angle = self.angle % -360


class city(object):

    def __init__(self, city_file):
        self.city_file = city_file
        self.city_img = cv.imread(self.city_file)
        (self.width, self.length, _) = self.city_img.shape

    def draw_car(
        self,
        x,
        y,
        width,
        height,
        angle,
        img,
        ):

        _angle = (180 - angle) * math.pi / 180.0
        b = math.cos(_angle) * 0.5
        a = math.sin(_angle) * 0.5
        pt0 = (int(x - a * height - b * width), int(y + b * height - a
               * width))
        pt1 = (int(x + a * height - b * width), int(y - b * height - a
               * width))
        pt2 = (int(2 * x - pt0[0]), int(2 * y - pt0[1]))
        pt3 = (int(2 * x - pt1[0]), int(2 * y - pt1[1]))
        pt4 = (int((pt0[0] + pt1[0]) / 2), int((pt0[1] + pt1[1]) / 2))
        pt5 = (int((pt0[0] + pt3[0]) / 2), int((pt0[1] + pt3[1]) / 2))
        pt6 = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))

        line_color = (200, 200, 200)
        line_thickness = 5

    # print(pt0, pt1, pt2, pt3, pt4)

        cv.line(img, pt2, pt3, line_color, line_thickness)
        cv.line(img, pt3, pt5, line_color, line_thickness)
        cv.line(img, pt6, pt2, line_color, line_thickness)
        cv.line(img, pt5, pt4, line_color, line_thickness)
        cv.line(img, pt6, pt4, line_color, line_thickness)

    def get_current_loc_map(
        self,
        x,
        y,
        size,
        angle=0,
        state=False,
        ):

        newcity_img = copy.deepcopy(self.city_img)

        if x - size / 2 < 0 or y - size / 2 < 0 or x + size / 2 \
            > self.length - 1 or y + size / 2 > self.width - 1:
            return np.ones((size, size, 3))
        else:
            y = self.width - y

        if state == True:
            img_crop = self.draw_car(
                x,
                y,
                20,
                10,
                angle,
                newcity_img,
                )

        img_crop = newcity_img[int(y - size / 2):int(y) + int(size / 2), int(x
                               - size / 2):int(x) + int(size / 2)]
        img_state = np.average(img_crop, axis=2) / 255
        return (img_crop, img_state)


class env(object):

    def __init__(
        self,
        car,
        city,
        city_map,
        car_img,
        ):
        self.car = car
        self.city = city
        self.city_map = city_map
        self.car_img = car_img
        self.car_img = cv.resize(self.car_img, (self.car.length,
                                 self.car.width))
        self.size = 80

    # car_x and car_y are center points of the car

    def show_image(self):
        newcity = copy.deepcopy(self.city)
        car_rotated = ndimage.rotate(self.car_img, self.car.angle)
        (car_wid, car_len, _) = car_rotated.shape
        pos_x = self.car.x - car_len // 2
        pos_y = newcity.width - (self.car.y - car_wid // 2)

        if pos_x < 0:
            pos_x = 0
        elif pos_x > newcity.length:
            pos_x = newcity.length - car_len

        if pos_y > newcity.width:
            pos_y = newcity.width - car_wid
        elif pos_y < 0:
            pos_y = 0

        car_rotated = cv.addWeighted(newcity.city_img[pos_y:pos_y
                + car_wid, pos_x:pos_x + car_len], 0.5, car_rotated, 1,
                0)
        newcity.city_img[pos_y:pos_y + car_wid, pos_x:pos_x
                         + car_len] = car_rotated
        return newcity

    def step(self, action):
        self.reward = 0
        self.velocity_x = 0.5
        self.velocity_y = 0
        done = False

        angle = math.radians(action)
        self.velocity_x = self.velocity_x * math.cos(angle) \
            - self.velocity_y * math.sin(angle)
        self.velocity_y = self.velocity_y * math.cos(angle) \
            + self.velocity_x * math.sin(angle)
        self.car.move(self.velocity_x, self.velocity_y, action)
        xx = self.goal_x - self.car.x
        yy = self.goal_y - self.car.y

        distance = np.sqrt((self.car.x - self.goal_x) ** 2 + (self.car.y
                           - self.goal_y) ** 2)
    
        car_loc, _ = self.city_map.get_current_loc_map(self.car.x,
                              self.car.y, self.size)
        sand_quality = np.sum(car_loc)
        sand_quality = sand_quality / (self.size * self.size * 3 * 255)
    
        # moving on the sand
        # check coordinates carefully image cordinate y is inverse of car coordinate y
    
        sand_check = np.sum(self.city_map.city_img[int(self.city.width
                            - self.car.y), int(self.car.x)]) / (255 * 3)
    
        if sand_check > 0:  # **** Check whether coords are correct
            self.reward = self.reward - 5.0
        else:
    
               # moving on the road
    
            self.reward = self.reward - 1.5
    
        if self.car.x - int(self.car.length / 2) < 5 or self.car.y \
            - int(self.car.width / 2) < 5 or self.car.x \
            - int(self.car.length / 2) > self.city_map.length - 5 \
            or self.car.y - int(self.car.width / 2) > self.city_map.width \
            - 5:
            self.boundary_hit_count = self.boundary_hit_count + 1
            self.reward = self.reward - 5.0
    
        if distance < self.last_distance:
            self.reward = self.reward + 5
        else:
            self.reward = self.reward + 2
    
        if distance < 25:
            self.reward = self.reward + 100
    
            self.goal_hit_count += 1
    
            if swap == 1:
                print ('Hit the Goal 2: (' + str(goal_x) + ', ' \
                    + str(goal_y) + ')')
                traversal_log.write('Train episode: '
                                    + str(train_episode_num)
                                    + ' Eval episode: '
                                    + str(eval_episode_num)
                                    + ' : Hit the Goal 2: (' + str(goal_x)
                                    + ', ' + str(goal_y) + ')\n')
                self.goal_x = 575
                self.goal_y = 530
                self.swap = 0
                done = True
            else:
                print ('Hit the Goal 1: (' + str(goal_x) + ', ' \
                    + str(goal_y) + ')')
                traversal_log.write('Train episode: '
                                    + str(train_episode_num)
                                    + ' Eval episode: '
                                    + str(eval_episode_num)
                                    + ' : Hit the Goal 1: (' + str(goal_x)
                                    + ', ' + str(goal_y) + ')\n')
                self.goal_x = 610
                self.goal_y = 45
                self.swap = 1
                done = True
    
        self.last_distance = distance
        self.current_step += 1
    
        img_crop = self.city.get_current_loc_map(self.car.x, self.car.y,
                self.car.angle, state=True)
        self.last_action = action
        self.last_reward = self.reward
    
        return (img_crop, self.reward, done)

    def reset(self):

        longueur = self.city.length
        largeur = self.city.width

        self.on_road_count = 0
        self.off_road_count = 0
        self.boundary_hit_count = 0
        self.goal_hit_count = 0
        self.episode_total_reward = 0.0
        self.reward = 0
        self.last_reward = 0
        self.last_action = 0
        self.goal_x = 575
        self.goal_y = 530
        self.swap = 0
        self.last_distance = 0
        self.current_step = 0

        self.car.angle = 0.0
        self.car.x = np.random.randint(100, longueur - 100)
        self.car.y = np.random.randint(100, largeur - 100)

