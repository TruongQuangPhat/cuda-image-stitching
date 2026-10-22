// Authors: HUỲNH TẤN PHƯỚC (23120334), pipeline integration.
// Sequential S1-S6 execution and per-stage timing.
// PPM is the intermediate format; Python handles JPEG decoding and PNG output.

#include "warping.h"
#include "common.h"
#include "versions.h"
#include "blending.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace wp = stitching::warping;
namespace bp = stitching::blending;
using Clock = std::chrono::steady_clock;
namespace {
double milliseconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
}  // namespace

int stitching::runV0(int argc, char** argv) {
    if (argc < 5 || argc > 8) {
        std::cerr << "Usage: v0_stitching a.ppm b.ppm panorama.ppm metrics.json "
                     "[feather|seam] [reprojection_threshold_px] [ground_truth_H.txt]\n";
        return 2;
    }
    try {
        const std::string mode = argc > 5 ? argv[5] : "feather";
        if (mode != "feather" && mode != "seam") throw std::runtime_error("Unknown blend mode");
        stitching::homography::HomographyConfig hc;
        if (argc > 6) hc.reprojection_threshold = std::stod(argv[6]);
        bp::BlendConfig bc;
        if (mode == "seam") {
            bc.mode = bp::BlendMode::MinimumErrorSeam;
            bc.seam_feather_radius = 8.0;
            bc.compensate_exposure = true;
        }
        const auto a = stitching::io::readPpm(argv[1]), b = stitching::io::readPpm(argv[2]);
        const auto begin = Clock::now();
        const auto ga = stitching::orb::toGrayscale(a.pixels.data(),a.width,a.height,3);
        const auto gb = stitching::orb::toGrayscale(b.pixels.data(),b.width,b.height,3);
        const auto pre = Clock::now();
        const auto detection_a = stitching::orb::detect(ga);
        const auto detection_b = stitching::orb::detect(gb);
        const auto s1 = Clock::now();
        const auto da = stitching::orb::describe(detection_a);
        const auto db = stitching::orb::describe(detection_b);
        const auto s2 = Clock::now();
        const auto matches = stitching::matching::match(da,db);
        const auto s3 = Clock::now();
        if (!matches.valid) throw std::runtime_error(matches.error_message);
        const auto h = stitching::homography::estimate(matches.matches,da,db,hc);
        const auto s4 = Clock::now();
        if (!h.valid) throw std::runtime_error(h.error_message);
        const auto warped = wp::warp(a,b,h);
        const auto s5 = Clock::now();
        if (!warped.valid) throw std::runtime_error(warped.error_message);
        const auto blended = bp::blend(warped,bc);
        const auto end = Clock::now();
        if (!blended.valid) throw std::runtime_error(blended.error_message);
        stitching::io::writePpm(argv[3],blended.panorama);
        std::ofstream metrics(argv[4]);
        if (!metrics) throw std::runtime_error("Cannot write metrics");
        metrics << std::setprecision(12) << "{\n\"success\":true,\n\"mode\":\"" << mode
            << "\",\n\"seam_applied\":" << (blended.seam_applied ? "true" : "false")
            << ",\n\"seam_feather_radius\":" << bc.seam_feather_radius
            << ",\n\"compensate_exposure\":" << (bc.compensate_exposure ? "true" : "false")
            << ",\n\"width\":" << a.width << ",\n\"height\":" << a.height
            << ",\n\"canvas_width\":" << blended.panorama.width
            << ",\n\"canvas_height\":" << blended.panorama.height
            << ",\n\"keypoints_a\":" << da.keypoints.size()
            << ",\n\"keypoints_b\":" << db.keypoints.size()
            << ",\n\"matches\":" << matches.matches.size() << ",\n\"inliers\":" << h.num_inliers
            << ",\n\"mean_reprojection_px\":" << h.mean_reprojection_error
            << ",\n\"ransac_threshold_px\":" << hc.reprojection_threshold
            << ",\n\"preprocess_ms\":" << milliseconds(begin,pre)
            << ",\n\"s1_ms\":" << milliseconds(pre,s1)
            << ",\n\"s2_ms\":" << milliseconds(s1,s2)
            << ",\n\"s3_ms\":" << milliseconds(s2,s3)
            << ",\n\"s4_ms\":" << milliseconds(s3,s4)
            << ",\n\"s5_ms\":" << milliseconds(s4,s5)
            << ",\n\"s6_ms\":" << milliseconds(s5,end)
            << ",\n\"total_ms\":" << milliseconds(begin,end) << ",\n\"H_B_to_A\":[";
        for (std::size_t i = 0; i < h.H.size(); ++i) metrics << (i ? "," : "") << h.H[i];
        metrics << "],\n\"exposure_gains\":[" << blended.exposure_gains[0] << ','
                << blended.exposure_gains[1] << ',' << blended.exposure_gains[2] << ']';
        if (argc > 7) {
            std::ifstream truth_file(argv[7]);
            std::array<double,9> truth{};
            for (double& value : truth)
                if (!(truth_file >> value) || !std::isfinite(value))
                    throw std::runtime_error("Invalid ground-truth matrix");
            double sum = 0, maximum = 0;
            constexpr int nx = 8, ny = 6;
            for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
                const double px = (x+0.5)*b.width/nx, py = (y+0.5)*b.height/ny;
                const auto expected = stitching::homography::project(truth,px,py);
                const auto actual = stitching::homography::project(h.H,px,py);
                const double error = std::hypot(expected.x-actual.x,expected.y-actual.y);
                sum += error; maximum = std::max(maximum,error);
            }
            metrics << ",\n\"holdout_mean_error_px\":" << sum/(nx*ny)
                    << ",\n\"holdout_max_error_px\":" << maximum;
        }
        metrics << "\n}\n";
        if (!metrics) throw std::runtime_error("Failed writing metrics");
        std::cout << mode << ": " << blended.panorama.width << 'x' << blended.panorama.height
                  << ", " << h.num_inliers << '/' << matches.matches.size() << " inliers, "
                  << milliseconds(begin,end) << " ms\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0 failed: " << error.what() << '\n';
        return 1;
    }
}

#ifndef STITCHING_NO_MAIN
int main(int argc, char** argv) { return stitching::runV0(argc,argv); }
#endif
