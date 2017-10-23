#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


def outlier_filter(cloud):
    print("running outlier filter")
    f = cloud.make_statistical_outlier_filter()
    f.set_mean_k(5)
    f.set_std_dev_mul_thresh(0.5)
    out_cloud = f.filter()
    return out_cloud

def voxel_downsampling(cloud):
    # Voxel grid downsampling of input cloud
    voxels = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    voxels.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    out_cloud = voxels.filter()
    print("Voxel grid downsampling {}".format(out_cloud.size))
    return out_cloud

def pass_through_filter(cloud):
    # filter through z axis first
    print("running pass through filter")
    pass_through = cloud.make_passthrough_filter()
    filter_axis = 'z'
    axis_min = 0.6
    axis_max = 1.
    pass_through.set_filter_field_name(filter_axis)
    pass_through.set_filter_limits(axis_min, axis_max)
    out_cloud = pass_through.filter()
    
    # filter through y axis
    pass_through = out_cloud.make_passthrough_filter()
    filter_axis = 'y'
    axis_min = -0.5
    axis_max = 0.5
    pass_through.set_filter_field_name(filter_axis)
    pass_through.set_filter_limits(axis_min, axis_max)
    out_cloud = pass_through.filter()
    return out_cloud

def ransac_filter(cloud):
    print("running ransac filter")
    segmenter = cloud.make_segmenter()
    segmenter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    segmenter.set_distance_threshold(max_distance)
    inliers, outliers = segmenter.segment()

    table_cloud = cloud.extract(inliers, negative=False)
    objects_cloud = cloud.extract(inliers, negative=True)

    # convert to ros format and publish
    objects_ros_data = pcl_to_ros(objects_cloud)
    table_ros_data = pcl_to_ros(table_cloud)

    pcl_objects_pub.publish(objects_ros_data)
    pcl_table_pub.publish(table_ros_data)

    return objects_cloud, table_cloud

def euclidean_clustering(cloud):
    print("Euclidean clustering")
    white_cloud = XYZRGB_to_XYZ(cloud)
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(2000)

    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []
 
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    return cluster_indices
    

def turn_pr2(pos, wait=True):
    time_elapsed = rospy.Time.now()
    pub_body.publish(pos)
    loc = 0.0

    joint_state = rospy.wait_for_message('/pr2/joint_states', JointState)
    loc = joint_state.position[19]	# the world link
        
    while wait:
        joint_state = rospy.wait_for_message('/pr2/joint_states', JointState)
        loc = joint_state.position[19]
 	if at_goal(loc, pos):
 	   #print "turn_pr2: Request: %f Joint %s=%f" % (pos, joint_state.name[19], joint_state.position[19])
           time_elapsed = joint_state.header.stamp - time_elapsed
           break

    return loc

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    pcl_cloud = ros_to_pcl(pcl_msg)

    cloud_filtered = voxel_downsampling(pcl_cloud)
    cloud_filtered = pass_through_filter(pcl_cloud)
    cloud_filtered = outlier_filter(pcl_cloud)
    cloud_filtered_objects, cloud_filtered_table = ransac_filter(cloud_filtered)  
    cluster_indices = euclidean_clustering(cloud_filtered_objects)


    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_filtered_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(cloud_filtered_objects[pts_list[0]])[0:3]
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Publish the list of detected objects

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects, cloud_filtered_table)
    except rospy.ROSInterruptException:
        pass


def join_clouds(c1, c2):
    assert c1.__class__() == c2.__class__()
    c3 = c1.__class__()
    c3.from_list(c1.to_list() + c2.to_list())
    return c3


def pr2_mover(object_list, table_cloud):
    dict_list = []

    test_scene_num = Int32()
    test_scene_num.data = 1

    # Get/Read the detected object parameters
    object_list_param = rospy.get_param('/object_list')

    # Loop through the pick list
    labels = []
    centroids = []  # to be list of tuples (x, y, z)
    scene_num = 1

    for object_to_pick in object_list_param:
        # Put the object parameters into individual variables
        pick_object_name = object_to_pick['name']
        pick_object_group = object_to_pick['group']
        print("Searching for {} to be placed in {}".format(pick_object_name, pick_object_group))

        object_name = String()
        object_name.data = str(pick_object_name)
        
        collision_cloud = table_cloud
        pick_object_cloud = None

        for i, detected_object in enumerate(object_list):
            if(detected_object.label == pick_object_name):
                pick_object_cloud = detected_object.cloud
            else:
                collision_cloud = join_clouds(collision_cloud, ros_to_pcl(detected_object.cloud))
        
        if(pick_object_cloud == None):
            print("Did not detect in object list".format(pick_object_name))
            continue
        
        collision_pub.publish(pcl_to_ros(collision_cloud))
        points_arr = ros_to_pcl(pick_object_cloud).to_array()
        pick_object_centroid = np.median(points_arr, axis=0)[:3] 
        print("Centroid found : {}".format(pick_object_centroid))

        labels.append(pick_object_name)
        centroids.append(pick_object_centroid)

        # Create pick_pose for the object
        pick_pose = Pose()
        pick_pose.position.x = float(pick_object_centroid[0])
        pick_pose.position.y = float(pick_object_centroid[1])
        pick_pose.position.z = float(pick_object_centroid[2])
       
                                                
        place_pose = Pose()
        if(pick_object_group == 'green'):
            place_pose.position.x =  0
            place_pose.position.y = -0.71
            place_pose.position.z =  0.605
        else:
            place_pose.position.x =  0
            place_pose.position.y =  0.71
            place_pose.position.z =  0.605
                
#        print("Place pose created,... {}")
                # Wait for 'pick_place_routine' service to come up
        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        if(pick_object_group == 'green'):
            arm_name.data = 'right'
        else:
            arm_name.data = 'left'
                    
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        
        dict_list.append(yaml_dict)
        
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            print("Creating service proxy,...")
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            print("Requesting for service reponse,...")
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)
            picked_objects.append(pick_object_name)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    # Output the request parameters into the enviroment output yaml file
    out_file = 'output_{}.yaml'.format(test_scene_num.data)
    send_to_yaml(out_file, dict_list)

if __name__ == '__main__':

    rospy.init_node('clustering', anonymous=True)

    # TODO: ROS node initialization

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    passthrough_pub = rospy.Publisher("/received_points", PointCloud2, queue_size = 1)
    collision_pub = rospy.Publisher("/pr2/3D_map/points", PointCloud2, queue_size = 1)
    joint_publisher = rospy.Publisher('/pr2/world_joint_controller/command',
                             Float64, queue_size=10)

    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    
    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
      rospy.spin()
