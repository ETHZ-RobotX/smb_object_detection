# the class name, e.g., 'person'
string class_id
int32 id 

# definition of the bounding box of the object in the image
int32 bounding_box_min_x
int32 bounding_box_min_y
int32 bounding_box_max_x
int32 bounding_box_max_y

# the confindence of the predictor
float64 confidence 

string pose_estimation_type

# position in 3D
geometry_msgs/Point position


