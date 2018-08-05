/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;
	num_particles = 100;

	// Create Normal Distributions for x,y,theta measurements
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	
	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.weight = 1.0;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);

		particles.push_back(particle);
		weights.push_back(1);

		}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//For each particle, update the position and heading based on the measurement + gaussion noise	
	
	default_random_engine gen;

	for(int i = 0 ; i < num_particles ; i++){

		double x_f, y_f, theta_f;

		if(fabs(yaw_rate) < 0.0001){
			x_f = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			y_f = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			theta_f = particles[i].theta;
		}

		else{
			x_f =  particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			y_f = particles[i].y + (velocity/yaw_rate)*( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			theta_f = particles[i].theta + yaw_rate*delta_t;
		}

		normal_distribution<double> dist_x(x_f,std_pos[0]);
		normal_distribution<double> dist_y(y_f,std_pos[1]);
		normal_distribution<double> dist_theta(theta_f,std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	// For each predicted landmark, find the transformed observation that is closest using a nested for-loop
	// The complexity of the below algorithm runs in O(m*n), where m is the number of observations and n
	// is the number of landmarks predicted within sensor range.

	double distance; // Stores distance between observation and predicted landmarks.
	double min_distance = numeric_limits<double>::max(); // Stores the smallest distance between observation and landmarks.
	int map_id = -1 ; //Stores the map ID of the closest landmark to each obesrvation

	for (unsigned int i = 0; i < observations.size(); i++){		
		
		LandmarkObs obs = observations[i];

		for (unsigned int j = 0; j< predicted.size(); j++){

			LandmarkObs pred = predicted[j];
			distance = dist (obs.x, obs.y , pred.x, pred.y );

			if(distance < min_distance){ 
				// This landmark is now the closest!
				map_id = pred.id;
				min_distance = distance;
			}
		}
		observations[i].id = map_id; // Update the ID of the observation to that of it's closest landmark!
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	//Loop across every particle
	for (int k = 0; k < num_particles; k++){

		double p_x, p_y, p_theta;

		//Extract particle position:
		p_x = particles[k].x;
		p_y = particles[k].y;
		p_theta = particles[k].theta;

		//Convert each landmark observation to the map's co-ordinate system:
		LandmarkObs obs; //Temporary variable to store each observation
		LandmarkObs trans_obs;  // Observation converted to map coordinate system
		vector<LandmarkObs> trans_observations; // List to store all transformed observations

		for (unsigned int i = 0 ; i < observations.size(); i ++){

			obs  = observations[i];

			trans_obs.x = p_x + obs.x*cos(p_theta) - obs.y*sin(p_theta);
			trans_obs.y = p_y + obs.x*sin(p_theta) + obs.y*cos(p_theta);
			trans_obs.id = obs.id;

			trans_observations.push_back(trans_obs);
		}

		// Only consider landmarks that are within the sensor range.
		// There is no point wasting computation for landmarks that are
		// past where the sensors can reach:
		vector<LandmarkObs> landmarks; // Stores all landmarks within range
		double distance; // Stores the distance between particle and landmark

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){

			double lm_x = map_landmarks.landmark_list[j].x_f;
      		double lm_y = map_landmarks.landmark_list[j].y_f;
      		int lm_id = map_landmarks.landmark_list[j].id_i;

      		distance = dist (lm_x ,lm_y, p_x, p_y);
      		if(distance < sensor_range){
      			landmarks.push_back( LandmarkObs{ lm_id, lm_x, lm_y } );
      		}
		}
		double trimmd = map_landmarks.landmark_list.size() - landmarks.size();
		//cout << " # of landmarks within range :  " << landmarks.size() << endl;

		// Associate each landmark with the closest measured observation
		dataAssociation(landmarks, trans_observations);

		//Update the particle weight by multiplying each measurements gaussian PDF
		particles[k].weight = 1.0;
		for (unsigned int i = 0;i<trans_observations.size(); i++){	

			double obs_x, obs_y, obs_id, pred_x, pred_y, s_x, s_y, obs_weight;
			obs_x = trans_observations[i].x;
			obs_y = trans_observations[i].y;
			obs_id = trans_observations[i].id;

			// Extract the associated landmark for the current observation:
			for (unsigned int j = 0; j < landmarks.size(); j++) {
				if (landmarks[j].id == obs_id) {
					pred_x = landmarks[j].x;
					pred_y = landmarks[j].y;
				}
			}
			// Compute the Gaussion Probability and update the weight.
			s_x = std_landmark[0];
      		s_y = std_landmark[1];

      		obs_weight = ( 1.0/(2.0*M_PI*s_x*s_y)) * exp(-(pow(obs_x - pred_x,2)/(2*s_x*s_x) + (pow(obs_y - pred_y,2)/(2*s_y*s_y))));

      		cout << "w = " << particles[k].weight << ". i = " << i << ;

       		particles[k].weight *= obs_weight;
		}
		cout << endl;
		weights[k] = particles[k].weight;
	}	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;

	discrete_distribution<int> distribution(weights.begin(),weights.end());
	vector<Particle> resample_particles;

	for (int i = 0; i< num_particles; i++){
		resample_particles.push_back(particles[distribution(gen)]);

	}
	particles = resample_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
