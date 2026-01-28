/************************************************************************
Copyright (c) 2018-2019, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/
#include "RobotState.h"
#include "stateEstimation.h"
#include "util.h"
#include "message.h"
#include "ROSnode.h"
#include "unitree_legged_msgs/LowCmd.h"
#include "unitree_legged_msgs/LowState.h"
#include "unitree_legged_msgs/HighState.h"
#include <std_msgs/Float64.h>
#include <std_msgs/Float32.h>
#include "ros/ros.h"
#include <stdio.h>
#include <stdlib.h>
#include "unitree_legged_msgs/MotorState.h"
#include <geometry_msgs/WrenchStamped.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <assert.h>
#include <Eigen/Dense>
#include "gazebo_msgs/ModelStates.h"
#include <fstream>
#include <time.h>
#include <ctime>
#include <chrono>
#include <thread>
#include <array>

using namespace Eigen;
using namespace std;

int main(int argc, char **argv)
{
    // std::thread _robot_ctl_thread(RobotCtl);
    // _robot_ctl_thread.join();

    //___________________________________________
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
  
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    const char* model_path = "/home/whale/JyRos/Sim2SimDeployment/src/robot_control/onnxmodels/bound_0723.onnx";
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> input_node_dims;

    bool dynamic_flag = false;

    for (int i = 0; i < num_input_nodes; i++) {
         //输出输入节点的名称
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // 输出输入节点的类型
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        input_node_dims = tensor_info.GetShape();
        //输入节点的打印维度
          printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        //打印各个维度的大小
        for (int j = 0; j < input_node_dims.size(); j++) 
        {
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
            if (input_node_dims[j] < 1)
            {
                dynamic_flag  = true;
            }
        }

        input_node_dims[0] = 1;
    }

//打印输出节点信息，方法类似
    for (int i = 0; i < num_output_nodes; i++)
    {
        char* output_name = session.GetOutputName(i, allocator);
        printf("Output: %d name=%s\n", i, output_name);
        output_node_names[i] = output_name;
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);
        auto output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }

    size_t input_obs_size = 1*49;  
    size_t output_action_size = 1*12;
    std::vector<float> input_obs_values(input_obs_size);

    std::vector<float> output_action_values(output_action_size);

    std::vector<int64_t> input_obs_dims;
    std::vector<int64_t> output_action_dims;


    Ort::TypeInfo input_obs_info = session.GetInputTypeInfo(0);
    auto input_obs_tensor_info = input_obs_info.GetTensorTypeAndShapeInfo();
    input_obs_dims = input_obs_tensor_info.GetShape();

    Ort::TypeInfo output_action_info = session.GetOutputTypeInfo(0);
    auto output_action_tensor_info = output_action_info.GetTensorTypeAndShapeInfo();
    output_action_dims = output_action_tensor_info.GetShape();
    
    //___________________________________________
    ros::init(argc, argv, "control");

    string robot_name;
    ros::param::get("/robot_name", robot_name);
    cout << "robot_name: " << robot_name << endl;

    multiThread listen_publish_obj(robot_name);
    // usleep(1000); // must wait 1ms, to get first state

    ros::NodeHandle n;
    ros::Publisher lowState_pub;
    ros::Publisher modelState_pub;

    std::thread _data_record_thread(SaveData);
    ros::AsyncSpinner spinner(12); // one threads
    spinner.start();

    lowState_pub = n.advertise<unitree_legged_msgs::LowState>("/" + robot_name + "_gazebo/lowState/state", 1);
    // modelState_pub = n.advertise<gazebo_msgs::ModelStatesConstPtr>("/gazebo/model_states", 1);
    servo_pub[0] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/FLL_joint_controller/command", 1);
    servo_pub[1] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/FLR_joint_controller/command", 1);
    servo_pub[2] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/FRL_joint_controller/command", 1);
    servo_pub[3] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/FRR_joint_controller/command", 1);
    servo_pub[4] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/HLL_joint_controller/command", 1);
    servo_pub[5] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/HLR_joint_controller/command", 1);
    servo_pub[6] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/HRL_joint_controller/command", 1);
    servo_pub[7] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/HRR_joint_controller/command", 1);
    servo_pub[8] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/FS_joint_controller/command", 1);
    servo_pub[9] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/HS_joint_controller/command", 1);
    servo_pub[10] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/LS_joint_controller/command", 1);
    servo_pub[11] = n.advertise<unitree_legged_msgs::MotorCmd>("/" + robot_name + "_gazebo/RS_joint_controller/command", 1);

    paramInit();
    // sendServoCmd();

    ros::Rate a_liite_time(5);
    ros::Duration one_second(1);
    extern float x_v, y_v, z_v;
    extern float quat_x, quat_y, quat_z, quat_w;
    int freq = 500;
    a_liite_time.sleep();
    

    int policy_freq = 50;
    float action[12]={0};
    float state[36]={0};
    float pos_init[12] = {0};

    ros::Rate loop_rate(freq);

    // ########## parameters setting #########################################################################################
    // #######################################################################################################################
    // #######################################################################################################################
    // #######################################################################################################################
    int spine_horizontal=0; // 1 for horizontal, 0 for vertical
    int gait_pattern=1; // 0 for trot, 1 for bound, 2 for pace, 3 for gallop
    float v_des=0.8;
    float command[3]={0};
    float yaw_des=1.0;
    float t_loop=0.3;
    float phase_time=t_loop*0.0;
    float kp_leg = 40;
    float kd_leg = 0.5;
    float kp_spine = 125;
    float kd_spine = 2.5;
    // ########################################################################################################################
    // ########################################################################################################################
    // ########################################################################################################################
    // ########################################################################################################################
    float pos_default[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if( spine_horizontal==1){
        pos_default[8] = 1.57;
        pos_default[9] = -1.57;
    }

    double pos_temp[12] ,lastPos[12], percent;
    for(int j=0; j<12; j++) lastPos[j] = lowState.motorState[j].q;
    
    for(int i=1; i<=freq; i++){
        if(!ros::ok()) break;
        percent = (double)i/freq;
        for(int j=0; j<12; j++){
            lowCmd.motorCmd[j].q = lastPos[j]*(1-percent) + pos_default[j]*percent; 
            lowCmd.motorCmd[j].Kp = 20+20*percent;
            lowCmd.motorCmd[j].Kd = 0.5+1*percent;
        }
        sendServoCmd();
        loop_rate.sleep();
    }

    float pos_des[12]={0};
    command[0]=v_des;
    command[1]=0;
    command[2]=yaw_des;

    robot_state.updateRobotState();


    get_observation(state,command);

    ros::Time prev = ros::Time::now();
    ros::Time now = ros::Time::now();
    ros::Duration dt(0);

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=MAIN  FUNCTION-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    while (ros::ok()){
        
        clock_t start, end;

        now = ros::Time::now();
        dt = now - prev;
        prev = now;
        command[0]=v_des;
        command[1]=0;
        command[2]=clip(yaw_des - robot_state.base_euler[2] ,-3.14,3.14);
        printf("yaw: %f \n", robot_state.base_euler[2]);
        // printf("yaw: %f \n", command[2]);
        get_observation(state,command);

        robot_state.updateRobotState();
        robot_state.detectContact();
        robot_state.upDateFootInfo();
        // std::cout << "dt: " << dt.toSec() << std::endl;
        robot_state_estimation.update_estimation(robot_state, dt.toSec());     //更新机器人状态
        
        lowState_pub.publish(lowState);
        
        static int num = 0;
        num++;
        
        if (abs(phase_time-t_loop) < ((float)0.5/freq))
            phase_time=0;
        // printf("phase time: %f \n", phase_time);
        // root_euler = quat_to_euler(quat_x,quat_y,quat_z,quat_w);

        for (unsigned int i = 0; i < 36; i++){
            input_obs_values[i] = state[i]; 
        }   
        for (unsigned int i = 0; i < 12; i++){
            input_obs_values[i+12] = state[i+12] - pos_default[i]; 
        } 

        for (unsigned int i = 0; i < 12; i++)
            input_obs_values[i+36] = action[i]; 

        input_obs_values[48] = phase_time;
        // input_obs_values[0]=0.8+0.1*sin(6.28*phase_time/t_loop);
        // input_obs_values[1]=0.1*sin(6.28*phase_time/t_loop);
        // input_obs_values[2]=1.25*cos(2*6.28*phase_time/t_loop)+0.25;
        
        // Run onnx model
        if(num==freq/policy_freq){
            // printf("len:%f \n",input_obs_values[10]);
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_obs_tensor = Ort::Value::CreateTensor<float>(memory_info, input_obs_values.data(), input_obs_values.size(), input_obs_dims.data(), 2);
            Ort::Value output_action_tensor = Ort::Value::CreateTensor<float>(memory_info, output_action_values.data(), output_action_values.size(), output_action_dims.data(), 2);
            std::vector<Ort::Value> ort_inputs;
            std::vector<Ort::Value> ort_outputs;
            assert(input_obs_tensor.IsTensor());
            assert(output_action_tensor.IsTensor());
            ort_inputs.push_back(std::move(input_obs_tensor));
            ort_outputs.push_back(std::move(output_action_tensor));
            
            // Ort::Value output_tensors=Ort::Value::CreateTensor<float>(memory_info, ort_outputs.data(), ort_outputs.size(), output_shape.data(), output_shape.size());
            // ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), output_node_names.size());
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), output_node_names.size());
            assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
      
            float* floatarr_action = output_tensors[0].GetTensorMutableData<float>();
            for (int i=0;i<12;i++)
            {
                action[i]=floatarr_action[i];
                // printf("action[%d]:%f \n",i,action[i]);
            }
            num=0;
        }
        //Run onnx model completed
        float secs =ros::Time::now().toSec();
        // printf(" out: %f \n",state[2]);
        
        // configure trajectory
        trajectory(phase_time,action,v_des,yaw_des,t_loop,gait_pattern);
        for(int j=0; j<12; j++) pos_des[j] = a_temp[j]+pos_default[j];
        // calculate torque
        // calculate_torque(pos_des, Kp_leg_phi, Kd_leg_phi, Kp_leg_len, Kd_leg_len, Kp_spine_phi, Kd_spine_phi, Kp_spine_len, Kd_spine_len);
        phase_time = phase_time + ((float)1/freq);
        // printf("phase time: %f \n", phase_time);
        
        // static std::array<float, 8> motor_pos = {0};
        // motor_pos[0] > -60.0 * 3.1415/180.0  ? motor_pos[0] -= 0.001 : motor_pos[0] = -60.0 * 3.1415/180.0;
        // motor_pos[1] <  60.0 * 3.1415/180.0  ? motor_pos[1] += 0.001 : motor_pos[1] =  60.0 * 3.1415/180.0;
        // motor_pos[2] <  60.0 * 3.1415/180.0  ? motor_pos[2] += 0.001 : motor_pos[2] =  60.0 * 3.1415/180.0;
        // motor_pos[3] > -60.0 * 3.1415/180.0  ? motor_pos[3] -= 0.001 : motor_pos[3] = -60.0 * 3.1415/180.0;
        // motor_pos[4] > -60.0 * 3.1415/180.0  ? motor_pos[4] -= 0.001 : motor_pos[4] = -60.0 * 3.1415/180.0;
        // motor_pos[5] <  60.0 * 3.1415/180.0  ? motor_pos[5] += 0.001 : motor_pos[5] =  60.0 * 3.1415/180.0;
        // motor_pos[6] <  60.0 * 3.1415/180.0  ? motor_pos[6] += 0.001 : motor_pos[6] =  60.0 * 3.1415/180.0;
        // motor_pos[7] > -60.0 * 3.1415/180.0  ? motor_pos[7] -= 0.001 : motor_pos[7] = -60.0 * 3.1415/180.0;

        for(int j=0; j<8; j++){
            lowCmd.motorCmd[j].q = pos_des[j];
            // lowCmd.motorCmd[j].q = motor_pos[j];
            // lowCmd.motorCmd[j].q = 0.0;
            lowCmd.motorCmd[j].Kp = kp_leg;
            lowCmd.motorCmd[j].Kd = kd_leg;
            }
        for(int j=8; j<12; j++){
            lowCmd.motorCmd[j].q = pos_des[j];
            // lowCmd.motorCmd[j].q = 0.0;
            lowCmd.motorCmd[j].Kp = kp_spine;
            lowCmd.motorCmd[j].Kd = kd_spine;
            }
        sendServoCmd();
        start = clock();
        loop_rate.sleep();  //keep loop rate
        end = clock();
        // std::cout <<"耗时："<<(double)(end - start) / CLOCKS_PER_SEC << std::endl;
    }
    _data_record_thread.join();
    return 0;
}


