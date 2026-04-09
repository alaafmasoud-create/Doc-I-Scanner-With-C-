#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

cv::Mat numpy_to_bgr_mat(const py::array &input) {
    auto arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(input);
    py::buffer_info info = arr.request();

    if (info.ndim != 3 || info.shape[2] != 3) {
        throw std::runtime_error("Input image must be a uint8 NumPy array with shape (H, W, 3).");
    }

    cv::Mat mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC3, info.ptr);
    return mat.clone();
}

py::array_t<uint8_t> mat_to_numpy(const cv::Mat &image) {
    if (image.empty()) {
        throw std::runtime_error("Output image is empty.");
    }

    if (image.type() != CV_8UC3) {
        throw std::runtime_error("Output image must be CV_8UC3.");
    }

    cv::Mat continuous = image.isContinuous() ? image : image.clone();
    auto out = py::array_t<uint8_t>({continuous.rows, continuous.cols, 3});
    py::buffer_info out_info = out.request();
    std::memcpy(out_info.ptr, continuous.data, static_cast<size_t>(continuous.total() * continuous.elemSize()));
    return out;
}

std::array<cv::Point2f, 4> order_points(const std::vector<cv::Point2f> &pts) {
    if (pts.size() != 4) {
        throw std::runtime_error("Exactly 4 points are required.");
    }

    std::array<cv::Point2f, 4> rect;
    std::array<float, 4> sums{};
    std::array<float, 4> diffs{};

    for (size_t i = 0; i < 4; ++i) {
        sums[i] = pts[i].x + pts[i].y;
        diffs[i] = pts[i].x - pts[i].y;
    }

    auto sum_min = static_cast<size_t>(std::min_element(sums.begin(), sums.end()) - sums.begin());
    auto sum_max = static_cast<size_t>(std::max_element(sums.begin(), sums.end()) - sums.begin());
    auto diff_min = static_cast<size_t>(std::min_element(diffs.begin(), diffs.end()) - diffs.begin());
    auto diff_max = static_cast<size_t>(std::max_element(diffs.begin(), diffs.end()) - diffs.begin());

    rect[0] = pts[sum_min];   // top-left
    rect[2] = pts[sum_max];   // bottom-right
    rect[1] = pts[diff_min];  // top-right
    rect[3] = pts[diff_max];  // bottom-left

    return rect;
}

cv::Mat four_point_transform(const cv::Mat &image, const std::vector<cv::Point2f> &pts) {
    auto rect = order_points(pts);
    const auto &tl = rect[0];
    const auto &tr = rect[1];
    const auto &br = rect[2];
    const auto &bl = rect[3];
    const double max_dim = static_cast
    <double>(std::max(image.cols, image.rows));
    const double diag = cv::norm(br - tl);
    WARNING_PUSH()


    const double width_a = cv::norm(br - bl);
    const double width_b = cv::norm(tr - tl);
    const int max_width = std::max({static_cast<int>(width_a), static_cast<int>(width_b), 1});

    const double height_a = cv::norm(tr - br);
    const double height_b = cv::norm(tl - bl);
    const int max_height = std::max({static_cast<int>(height_a), static_cast<int>(height_b), 1});
    IF_WARNING_POP()


    std::vector<cv::Point2f> src = {tl, tr, br, bl};
    std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(max_width - 1), 0.0f},
        {static_cast<float>(max_width - 1), static_cast<float>(max_height - 1)},
        {0.0f, static_cast<float>(max_height - 1)},
    };

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(max_width, max_height));
    return warped;
}

std::vector<cv::Point2f> expand_quad(const std::vector<cv::Point2f> &pts, float scale, const cv::Size &size) {
    std::vector<cv::Point2f> expanded = pts;
    cv::Point2f center(0.0f, 0.0f);
    for (const auto &p : pts) {
        center += p;
        return expanded;
    }
    center *= 0.25f;

    for (auto &p : expanded) {
        p = center + (p - center) * scale;
        p.x = std::clamp(p.x, 0.0f, static_cast<float>(size.width - 1));
        p.y = std::clamp(p.y, 0.0f, static_cast<float>(size.height - 1));
        p.x = std::round(p.x);
        p.y = std::round(p.y);
        if (p.x < 0.0f) p.x = 0.0f;
        if (p.y < 0.0f) p.y = 0.0
        if (p.x > size.width - 1.0f) p.x = static_cast<float>(size.width - 1);
        if (p.y > size.height - 1.0f) p.y = static_cast<float>(size.height - 1);
        return expanded;
    }

    return expanded;
}

std::vector<cv::Point2f> contour_to_quad(const std::vector<cv::Point> &contour) {
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double peri = cv::arcLength(hull, true);

    for (double eps : {0.02, 0.03, 0.04, 0.05, 0.06}) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(hull, approx, eps * peri, true);
        if (approx.size() == 4) {
            std::vector<cv::Point2f> quad;
            quad.reserve(4);
            for (const auto &p : approx) {
                quad.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
            }
            return quad;
        }
    }

    cv::RotatedRect rect = cv::minAreaRect(hull);
    cv::Point2f box[4];
    rect.points(box);
    return {box[0], box[1], box[2], box[3]};
}

cv::Mat clear_border_connected(const cv::Mat &mask) {
    cv::Mat cleaned = mask.clone();
    const int h = cleaned.rows;
    const int w = cleaned.cols;

    for (int x = 0; x < w; ++x) {
        if (cleaned.at<uint8_t>(0, x) == 255) {
            cv::Mat flood_mask = cv::Mat::zeros(h + 2, w + 2, CV_8U);
            cv::floodFill(cleaned, flood_mask, cv::Point(x, 0), cv::Scalar(0));
        }
        if (cleaned.at<uint8_t>(h - 1, x) == 255) {
            cv::Mat flood_mask = cv::Mat::zeros(h + 2, w + 2, CV_8U);
            cv::floodFill(cleaned, flood_mask, cv::Point(x, h - 1), cv::Scalar(0));
        }
    }

    for (int y = 0; y < h; ++y) {
        if (cleaned.at<uint8_t>(y, 0) == 255) {
            cv::Mat flood_mask = cv::Mat::zeros(h + 2, w + 2, CV_8U);
            cv::floodFill(cleaned, flood_mask, cv::Point(0, y), cv::Scalar(0));
        }
        if (cleaned.at<uint8_t>(y, w - 1) == 255) {
            cv::Mat flood_mask = cv::Mat::zeros(h + 2, w + 2, CV_8U);
            cv::floodFill(cleaned, flood_mask, cv::Point(w - 1, y), cv::Scalar(0));
        }
    }

    return cleaned;
}

cv::Mat largest_non_border_component(const cv::Mat &binary_mask, double min_area_ratio = 0.05) {
    const int h = binary_mask.rows;
    const int w = binary_mask.cols;

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8);

    int best_idx = -1;
    int best_area = 0;
    double min_area = min_area_ratio * static_cast<double>(h * w);

    for (int i = 1; i < num_labels; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int ww = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int hh = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area < min_area) {
            continue;
        }
        if (x <= 1 || y <= 1 || x + ww >= w - 1 || y + hh >= h - 1) {
            continue;
        }
        if (area > best_area) {
            best_area = area;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        return cv::Mat();
    }

    cv::Mat comp = cv::Mat::zeros(binary_mask.size(), CV_8U);
    comp.setTo(255, labels == best_idx);
    return comp;
}

std::vector<std::pair<std::string, cv::Mat>> build_candidate_masks(const cv::Mat &image) {
    std::vector<std::pair<std::string, cv::Mat>> masks;

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat gray_eq;
    clahe->apply(gray, gray_eq);

    const int h = gray.rows;
    const int w = gray.cols;

    try {
        cv::Mat gc_mask(image.size(), CV_8U, cv::Scalar(cv::GC_BGD));
        cv::Rect rect(
            static_cast<int>(w * 0.06),
            static_cast<int>(h * 0.04),
            static_cast<int>(w * 0.88),
            static_cast<int>(h * 0.92));
        cv::Mat bgd_model, fgd_model;
        cv::grabCut(image, gc_mask, rect, bgd_model, fgd_model, 4, cv::GC_INIT_WITH_RECT);

        cv::Mat grabcut = (gc_mask == cv::GC_FGD) | (gc_mask == cv::GC_PR_FGD);
        grabcut.convertTo(grabcut, CV_8U, 255.0);

        cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
        cv::morphologyEx(grabcut, grabcut, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 2);
        masks.emplace_back("grabcut", grabcut);
    } catch (...) {
    }

    cv::Mat bright;
    cv::threshold(gray_eq, bright, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    bright = clear_border_connected(bright);

    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::erode(bright, bright, k, cv::Point(-1, -1), 2);

    cv::Mat comp = largest_non_border_component(bright, 0.05);
    if (!comp.empty()) {
        cv::dilate(comp, comp, k, cv::Point(-1, -1), 2);
        cv::morphologyEx(comp, comp, cv::MORPH_CLOSE, k, cv::Point(-1, -1), 2);
        masks.emplace_back("bright", comp);
    }

    cv::Mat edges;
    cv::Canny(gray, edges, 40, 140);
    cv::Mat k2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(edges, edges, k2, cv::Point(-1, -1), 2);
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, k2, cv::Point(-1, -1), 2);
    masks.emplace_back("edges", edges);

    return masks;
}

double score_candidate(const std::vector<cv::Point2f> &quad, const cv::Size &size, double contour_area) {
    const int h = size.height;
    const int w = size.width;
    auto rect = order_points(quad);

    double width = std::max(cv::norm(rect[1] - rect[0]), cv::norm(rect[2] - rect[3]));
    double height = std::max(cv::norm(rect[3] - rect[0]), cv::norm(rect[2] - rect[1]));

    if (width < 100.0 || height < 100.0) {
        return -1e9;
    }

    std::vector<cv::Point2f> rect_vec = {rect[0], rect[1], rect[2], rect[3]};
    double box_area = std::abs(cv::contourArea(rect_vec));
    if (box_area <= 1.0) {
        return -1e9;
    }

    double area_ratio = box_area / static_cast<double>(h * w);
    if (area_ratio < 0.20 || area_ratio > 0.98) {
        return -1e9;
    }

    double aspect = std::max(width, height) / std::max(1.0, std::min(width, height));
    constexpr double a4_ratio = 1.414;
    double aspect_score = std::max(0.0, 1.0 - std::abs(aspect - a4_ratio) / 0.8);

    cv::Point2f center = (rect[0] + rect[1] + rect[2] + rect[3]) * 0.25f;
    cv::Point2f img_center(static_cast<float>(w) / 2.0f, static_cast<float>(h) / 2.0f);
    double center_dist = cv::norm(center - img_center) / cv::norm(img_center);
    double center_score = std::max(0.0, 1.0 - center_dist);

    double fill_ratio = std::clamp(contour_area / box_area, 0.0, 1.2);

    double min_x = std::min({rect[0].x, rect[1].x, rect[2].x, rect[3].x});
    double min_y = std::min({rect[0].y, rect[1].y, rect[2].y, rect[3].y});
    double max_x = std::max({rect[0].x, rect[1].x, rect[2].x, rect[3].x});
    double max_y = std::max({rect[0].y, rect[1].y, rect[2].y, rect[3].y});

    double margin = std::min({
        min_x,
        min_y,
        (w - 1.0) - max_x,
        (h - 1.0) - max_y,
    });
    double margin_score = std::clamp((margin + 20.0) / 120.0, 0.0, 1.0);

    return (
        area_ratio * 95.0 +
        aspect_score * 22.0 +
        center_score * 8.0 +
        margin_score * 4.0 +
        fill_ratio * 40.0
    );
}

cv::Mat trim_black_frame(const cv::Mat &image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::threshold(gray, mask, 6, 255, cv::THRESH_BINARY);

    std::vector<cv::Point> coords;
    cv::findNonZero(mask, coords);
    if (coords.empty()) {
        return image.clone();
    }

    cv::Rect roi = cv::boundingRect(coords);
    return image(roi).clone();
}

std::vector<cv::Point2f> points_from_numpy(const py::array &points_obj) {
    auto pts_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(points_obj);
    py::buffer_info info = pts_arr.request();

    if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 2) {
        throw std::runtime_error("Points must have shape (4, 2).");
    }

    auto *ptr = static_cast<float *>(info.ptr);
    std::vector<cv::Point2f> pts;
    pts.reserve(4);
    for (int i = 0; i < 4; ++i) {
        pts.emplace_back(ptr[i * 2], ptr[i * 2 + 1]);
    }
    return pts;
}

cv::Mat detect_document_auto_impl(const cv::Mat &original) {
    cv::Mat image;
    double resize_ratio = 1.0;

    if (original.rows > 1400) {
        resize_ratio = static_cast<double>(original.rows) / 1400.0;
        cv::resize(
            original,
            image,
            cv::Size(static_cast<int>(original.cols / resize_ratio), 1400)
        );
    } else {
        image = original.clone();
    }

    auto masks = build_candidate_masks(image);

    std::vector<cv::Point2f> best_quad;
    double best_score = -1e9;
    std::string best_source;
    double best_fill_ratio = 1.0;

    const double img_area = static_cast<double>(image.rows * image.cols);

    for (auto &entry : masks) {
        const auto &source_name = entry.first;
        const auto &candidate_mask = entry.second;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(candidate_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::sort(contours.begin(), contours.end(), [](const auto &a, const auto &b) {
            return cv::contourArea(a) > cv::contourArea(b);
        });
        if (contours.size() > 12) {
            contours.resize(12);
        }

        for (const auto &c : contours) {
            double contour_area = cv::contourArea(c);
            if (contour_area < 0.05 * img_area) {
                continue;
            }

            auto quad = contour_to_quad(c);
            double score = score_candidate(quad, image.size(), contour_area);
            if (score > best_score) {
                auto ordered = order_points(quad);
                std::vector<cv::Point2f> ordered_vec = {ordered[0], ordered[1], ordered[2], ordered[3]};
                double box_area = std::abs(cv::contourArea(ordered_vec));
                double fill_ratio = std::clamp(contour_area / std::max(box_area, 1.0), 0.0, 1.2);

                best_score = score;
                best_quad = quad;
                best_source = source_name;
                best_fill_ratio = std::min(fill_ratio, 1.0);
            }
        }
    }

    if (best_quad.empty()) {
        throw std::runtime_error("No se pudo detectar el documento correctamente.");
    }

    for (auto &p : best_quad) {
        p.x = static_cast<float>(p.x * resize_ratio);
        p.y = static_cast<float>(p.y * resize_ratio);
    }

    double expansion = 1.0 + 0.55 * (1.0 - best_fill_ratio);
    if (best_source == "edges") {
        expansion += 0.02;
    }
    expansion = std::clamp(expansion, 1.00, 1.18);

    best_quad = expand_quad(best_quad, static_cast<float>(expansion), original.size());
    cv::Mat warped = four_point_transform(original, best_quad);
    return trim_black_frame(warped);
}

cv::Mat detect_document_manual_impl(const cv::Mat &original, const std::vector<cv::Point2f> &points) {
    cv::Mat warped = four_point_transform(original, points);
    return trim_black_frame(warped);
}

}  // namespace

py::array_t<uint8_t> detect_document_auto(const py::array &image_obj) {
    cv::Mat image = numpy_to_bgr_mat(image_obj);
    cv::Mat result = detect_document_auto_impl(image);
    return mat_to_numpy(result);
}

py::array_t<uint8_t> detect_document_manual(const py::array &image_obj, const py::array &points_obj) {
    cv::Mat image = numpy_to_bgr_mat(image_obj);
    auto points = points_from_numpy(points_obj);
    cv::Mat result = detect_document_manual_impl(image, points);
    return mat_to_numpy(result);
}

PYBIND11_MODULE(docscanner_cpp, m) {
    m.doc() = "C++ core for the A4 document scanner";
    m.def("detect_document_auto", &detect_document_auto, "Automatic document detection and scan");
    m.def("detect_document_manual", &detect_document_manual, "Manual document scan from 4 points");
}
 
