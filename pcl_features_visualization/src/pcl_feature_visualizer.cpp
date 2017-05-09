#include <ros/ros.h>
#include <ros/package.h>

#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>

#include <dynamic_reconfigure/server.h>
#include <pcl_features_visualization/PclFeaturesConfig.h>

#define NODE_NAME "pcl_feature_visualizer"
typedef pcl::PointXYZRGB PointType;

typedef struct
{
	volatile bool changed;
	double normalEstimationMaxDepthChangeFactor;
	double normalEstimationSmoothingSize;
	int pfhNumberOfBins;
	double pfhSearchRadius;
} Config;

Config config;

void dynamicReconfigureCallback(
		pcl_features_visualization::PclFeaturesConfig &cfg, uint32_t level)
{
	config.changed = true;
	config.normalEstimationMaxDepthChangeFactor =
			cfg.normal_estimation_max_depth_change_factor;
	config.normalEstimationSmoothingSize = cfg.normal_estimation_smoothing_size;
	config.pfhNumberOfBins = cfg.pfh_number_of_bins;
	config.pfhSearchRadius = cfg.pfh_search_radius;
	/* This is unnecessary for organized point clouds
	if (config.pfhSearchRadius > config.normalEstimationSearchRadius)
	{
		ROS_WARN("PFH Search radius should be greater than normal estimation search radius");
	}
	*/
}

class PclFeatureVisualizer
{
private:
	ros::NodeHandle n_;

	pcl::PointCloud<PointType>::Ptr cloud_;
	pcl::PointCloud<pcl::Normal>::Ptr normals_;

	pcl::visualization::PCLVisualizer viewer_;
	pcl::visualization::PCLHistogramVisualizer histogramViewer_;

	int selectedPointIndex_;
	PointType selectedPoint_;

	ros::Timer visualizerUpdateTimer_;
	ros::Timer changeCheckTimer_;

public:
	PclFeatureVisualizer(const pcl::PointCloud<PointType>& cloud) :
			viewer_("PCL Viewer"),
			cloud_(new pcl::PointCloud<PointType>(cloud)),
			selectedPointIndex_(-1)
	{
		viewer_.setBackgroundColor(0.7, 0.7, 0.7);
		recomputeNormals();
		draw();
		viewer_.registerPointPickingCallback(&PclFeatureVisualizer::pointPickInVisualizer, *this);
		visualizerUpdateTimer_ = n_.createTimer(ros::Duration(0.1), &PclFeatureVisualizer::updateVisualizer, this);
		changeCheckTimer_ = n_.createTimer(ros::Duration(0.1), &PclFeatureVisualizer::checkForChange, this);
	}

	void recomputeNormals()
	{
		normals_.reset(new pcl::PointCloud<pcl::Normal>());

		pcl::IntegralImageNormalEstimation<PointType, pcl::Normal> ne;
		ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
		ne.setMaxDepthChangeFactor(config.normalEstimationMaxDepthChangeFactor);
		ne.setNormalSmoothingSize(config.normalEstimationSmoothingSize);
		ne.setInputCloud(cloud_);
		ne.compute(*normals_);
	}

	void draw()
	{
		viewer_.removeAllPointClouds();
		pcl::visualization::PointCloudColorHandlerRGBField<PointType> colorHandler(cloud_);
		viewer_.addPointCloud<PointType>(cloud_, colorHandler, "cloud");
		viewer_.addPointCloudNormals<PointType, pcl::Normal>(cloud_, normals_, 20, 0.03, "normals");
	}

	void redraw()
	{
		pcl::visualization::PointCloudColorHandlerRGBField<PointType> colorHandler(cloud_);
		viewer_.updatePointCloud<PointType>(cloud_, colorHandler, "cloud");
		viewer_.removePointCloud("normals");
		viewer_.addPointCloudNormals<PointType, pcl::Normal>(cloud_, normals_, 20, 0.03, "normals");
	}

	void updateVisualizer(const ros::TimerEvent &ev)
	{
		viewer_.spinOnce();
		histogramViewer_.spinOnce();
	}

	void checkForChange(const ros::TimerEvent &ev)
	{
		if (config.changed)
		{
			config.changed = false;
			recomputeNormals();
			redraw();
		}
	}

	void pointPickInVisualizer(const pcl::visualization::PointPickingEvent& event, void*)
	{
		// int index = event.getPointIndex(); -> BAD! This returns the index if cloud has NO NaNs
		// http://docs.pointclouds.org/trunk/classpcl_1_1visualization_1_1_point_picking_event.html#ae835514e2744ed728c6de6d0b1f3b489
		pcl::search::KdTree<PointType> search;
		std::vector<int> indices(1);
		std::vector<float> distances(1);
		search.setInputCloud(cloud_);
		PointType pickedPoint;
		event.getPoint(pickedPoint.x, pickedPoint.y, pickedPoint.z);
		search.nearestKSearch(pickedPoint, 1, indices, distances);

		int index = indices[0];

		if (selectedPointIndex_ >= 0) // Restore the color to the previously selected point
		{
			cloud_->at(selectedPointIndex_) = selectedPoint_;
		}

		PointType& p = cloud_->at(index);
		selectedPoint_ = p; // save before changing the color

		p.r = 255;
		p.g = 0;
		p.b = 0;

		selectedPointIndex_ = index;

		computePFH();
		redraw();
	}

	void computePFH()
	{
		if (!pcl::isFinite<pcl::Normal>(normals_->points[selectedPointIndex_]))
		{
			PCL_WARN("normals[%d] is not finite\n", selectedPointIndex_);
			return;
		}
		pcl::PointCloud<PointType>::Ptr keypoints(new pcl::PointCloud<PointType>());
		keypoints->push_back(selectedPoint_);

		pcl::PFHEstimation<PointType, pcl::Normal, pcl::PFHSignature125> pfh;
		pfh.setSearchSurface(cloud_);
		pfh.setInputCloud(keypoints);
		pfh.setInputNormals(normals_);

		pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType> ());
		pfh.setSearchMethod(tree);
		pfh.setRadiusSearch(config.pfhSearchRadius);

		pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());
		pfh.compute(*pfhs);

		if (!histogramViewer_.updateFeatureHistogram(*pfhs, 125, "pfh histogram"))
		{
			std::cout << "Ignore the previous warning... it's ok\n";
			std::cout.flush();
			histogramViewer_.addFeatureHistogram(*pfhs, 125, "pfh histogram", 800, 600);
		}
	}
};

int main(int argc, char* argv[])
{
	ros::init(argc, argv, NODE_NAME);
	ros::NodeHandle n;

	dynamic_reconfigure::Server<pcl_features_visualization::PclFeaturesConfig> server;
	dynamic_reconfigure::Server<pcl_features_visualization::PclFeaturesConfig>::CallbackType f;
	f = boost::bind(&dynamicReconfigureCallback, _1, _2);
	server.setCallback(f);

	std::string filename = ros::package::getPath("pcl_features_visualization")
			+ "/pcds/cloud.pcd";

	pcl::PointCloud<PointType>::Ptr cloud(
			new pcl::PointCloud<PointType>);

	if (pcl::io::loadPCDFile(filename, *cloud) < 0)
	{
		ROS_ERROR("Could not read the pcd file %s", filename.c_str());
		return -1;
	}

	PclFeatureVisualizer visualizer(*cloud);

	ros::spin();
	return 0;
}
