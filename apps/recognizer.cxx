#include <ultimate_base_api_private.h>
#include <ultimate_text_api_private.h>
#include <compv/compv_api.h>
#include "../../ultimate_micr_tesseract_params.h"

#include <sstream>

using namespace COMPV_NAMESPACE;
using namespace ULTIMATE_BASE_NAMESPACE;
using namespace ULTIMATE_TEXT_NAMESPACE;

#define TAG_SAMPLE			"Recognizer Sample App"

#define OCR_MIN_CONFIDENCE		0.3			// Ignore any OCR result with a confidence value less than "MIN_CONFIDENCE"
#define OCR_OUT_FILE_NAME		"ocr.txt"   // name of file where to write OCR results [Single line seperated by comma for each file]
#define OCR_CRLF				"\r\n"		// End-Of-Line

#define OCR_THREADS_COUNT		COMPV_NUM_THREADS_MULTI // CPU-multithreading
#define OCR_GPGPU_ENABLED		true // Whether to enable OpenCL for prediction

int main(int argc, char** argv)
{
	{
		// Change debug level to INFO before starting to see what's going on
		CompVDebugMgr::setLevel(COMPV_DEBUG_LEVEL_INFO);

		// Print usage
		COMPV_DEBUG_INFO_EX(TAG_SAMPLE, "*** Usage: tesseract_recognizer.exe path_to_images_folder path_to_tessdata_folder ***");

		// Check arguments
		if (argc != 3) {
			COMPV_DEBUG_ERROR_EX(TAG_SAMPLE, "This application accepts a single param: list of images in txt file");
			COMPV_CHECK_CODE_RETURN(COMPV_ERROR_CODE_E_INVALID_PARAMETER, "Invalid number of parameters. Expected single one.");
		}

		// Init the modules
		COMPV_CHECK_CODE_RETURN(UltBaseEngine::init(OCR_THREADS_COUNT, OCR_GPGPU_ENABLED), "Failed to initialize the engine");

		// Make sure the provided arguments are valid txt file and contains a list of images
		const char* path_to_images_folder = argv[1];
		const char* path_to_tessdata_folder = argv[2];
		COMPV_DEBUG_INFO_EX(TAG_SAMPLE, "path_to_images_folder: %s", path_to_images_folder);
		COMPV_DEBUG_INFO_EX(TAG_SAMPLE, "path_to_tessdata_folder: %s", path_to_tessdata_folder);
		if (!CompVFileUtils::exists(path_to_images_folder)) {
			COMPV_DEBUG_ERROR_EX(TAG_SAMPLE, "[%s] doesn't exist", path_to_images_folder);
			COMPV_CHECK_CODE_RETURN(COMPV_ERROR_CODE_E_FILE_NOT_FOUND);
		}
		std::string path_to_tessdata_folder_str = path_to_tessdata_folder;
		if (path_to_tessdata_folder_str.back() != '/') {
			path_to_tessdata_folder_str += "/";
		}
		std::string path_to_images_folder_str = path_to_images_folder;
		if (path_to_images_folder_str.back() != '/') {
			path_to_images_folder_str += "/";
		}
		const std::string path_to_tessdata_model = path_to_tessdata_folder_str + "e13b.traineddata";
		if (!CompVFileUtils::exists(path_to_tessdata_model.c_str())) {
			COMPV_DEBUG_ERROR_EX(TAG_SAMPLE, "[%s] doesn't exist", path_to_tessdata_model.c_str());
			COMPV_CHECK_CODE_RETURN(COMPV_ERROR_CODE_E_FILE_NOT_FOUND);
		}

		// Get files in "images" folder
		std::vector<std::string> files;
		COMPV_CHECK_CODE_ASSERT(CompVFileUtils::getFilesInDir(path_to_images_folder_str.c_str(), files), "Failed to list of images in folder");
		files.erase(std::remove_if(files.begin(), files.end(), [](const std::string& path) {
			const std::string ext = CompVFileUtils::getExt(path.c_str());
			return (ext != "JPG" && ext != "JPEG" && ext != "PNG" && ext != "GIF" && ext != "BMP");
		}), files.end());

		// Make sure there are at least #1 file
		COMPV_CHECK_EXP_RETURN(files.empty(), COMPV_ERROR_CODE_E_INVALID_CALL, "No entry");

		// Layers
		UltBaseToggleMappingLayersVector layers = ultmicr_tesseract_params_build_layers();
		UltTextConfig config = ultmicr_tesseract_params_build_config();
		config.ocr_models_folder = path_to_tessdata_folder_str.c_str();
		config.visual_debug_enabled = false;
		config.ocr_patch_debug_save = false;

		// Guess number of threads to use. Should be the same as the number of virtual cores (most likely #8).
		CompVThreadDispatcherPtr threadDisp = CompVParallel::threadDispatcher();
		const size_t maxThreads = threadDisp ? static_cast<size_t>(threadDisp->threadsCount()) : 1; // number of virtual cores
		const size_t threadsCount = (threadDisp && !threadDisp->isMotherOfTheCurrentThread())
			? CompVThreadDispatcher::guessNumThreadsDividingAcrossY(1, files.size(), maxThreads, 1)
			: 1;
		COMPV_DEBUG_INFO_EX(TAG_SAMPLE, "Number of threads to use: %zu", threadsCount);

		// Create recognizer
		UltTextRecognizerPtr recognizer;
		COMPV_CHECK_CODE_RETURN(UltTextRecognizer::newObj(&recognizer, config), "Failed to create recognizer object");

		// Overrided context for this project
		UltTextConfigOverrided overridedConfig(config);
		overridedConfig.group_max_neighbs_interspace_scale = TESSERACT_CONFIG_FUSER_MAX_NEIGHBS_INTERSPACE_SCALE;

		// Run batches using thread dispatcher
		volatile size_t progress = 0;
		std::vector<std::string> predictions(files.size(), "");
		auto funcPtr = [&](const size_t start, const size_t end, const size_t threadIdx) -> COMPV_ERROR_CODE {
			COMPV_ASSERT(threadIdx < threadsCount);
			CompVMatPtr mt_image;
			UltTextGroups mt_groups;
			UltTextContext mt_context(layers, &config);
			std::vector<std::string> mt_texts;
			for (size_t i = start; i < end; ++i) {
				const std::string& file_path = files[i];
				std::string& prediction = predictions[i];
				const std::string file_name = CompVFileUtils::getFileNameFromFullPath(file_path.c_str());
				COMPV_DEBUG_INFO_EX(TAG_SAMPLE, "*** Thread %zu: Processing file '%s -> %s' [%zu/%zu] ***", threadIdx,
					file_name.c_str(), file_path.c_str(),
					compv_atomic_inc(&progress), files.size());
				
				// Decode the image
				COMPV_CHECK_CODE_ASSERT(CompVImage::decode(file_path.c_str(), &mt_image), "Failed to decode image");

				// OCR'ing
				COMPV_CHECK_CODE_ASSERT(recognizer->classifier()->process(mt_context, mt_image, mt_groups), "Classifier failed"); // Build candidate groups and classify (MICR/nonMICR)
				if (!mt_groups.groups.empty()) {
					COMPV_CHECK_CODE_ASSERT(UltTextFuser::process(mt_context, mt_groups, recognizer->grouper(), &overridedConfig), "Group fusion failed"); // Fuse the multi-layer groups
					COMPV_CHECK_CODE_ASSERT(recognizer->process(mt_context, mt_groups), "OCR'ing failed"); // Actual Tesseract OCR'ing

					// Sort from left to right and concat
					std::sort(mt_groups.groups.begin(), mt_groups.groups.end(), [](const UltTextGroup &a, const UltTextGroup &b) { return a.box.left < b.box.left; }); // left -> right
					for (const auto& it : mt_groups.groups) {
						if (it.ocrData && it.ocrData->confidence >= OCR_MIN_CONFIDENCE) {
							prediction += std::string(prediction.empty() ? "" : ",") + it.ocrData->utf8_string;
						}
					}
				}
				prediction = file_name + ": " + prediction + OCR_CRLF;
			}
			return COMPV_ERROR_CODE_S_OK;
		};
		COMPV_CHECK_CODE_ASSERT(CompVThreadDispatcher::dispatchDividingAcrossY(
			funcPtr,
			files.size(),
			threadsCount
		), "Thread dispatcher failed to start");

		// Flatten predictions and write to output file
		std::string predictions_flat;
		for (const auto& it : predictions) {
			predictions_flat += it;
		}
		COMPV_CHECK_CODE_ASSERT(CompVFileUtils::write(OCR_OUT_FILE_NAME, predictions_flat.c_str(), predictions_flat.size()), "Failed to write predictions to output file");
	}
	
	COMPV_CHECK_CODE_ASSERT(UltBaseEngine::deInit(), "Failed to unload the engine");
	COMPV_DEBUG_CHECK_FOR_MEMORY_LEAKS();

	/* DONE! */
	COMPV_DEBUG_INFO_EX(TAG_SAMPLE, "*** Press any key to terminate !! ***");
	getchar();

	return 0;
}
