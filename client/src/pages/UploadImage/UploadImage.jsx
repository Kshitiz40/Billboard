import React, { useEffect, useState } from "react";
import FileUploader from "../../components/FileUploader";
import Button from "../../components/Button";
import { useNotification } from "../../context/NotificationContext";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import Error from "../../components/Error";
import Sidebar from "../../components/Sidebar";
import { uploadImageToCloudinary } from './../../config/uploadToCloudinary';
import apibaseurl from '../../apiConfig/api';

const UploadImage = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [billboard, setBillboard] = useState(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [banner, setBanner] = useState(null);
  const [currentStep, setCurrentStep] = useState(1); // Step 1: Billboard, Step 2: Banner
  const { showNotification } = useNotification();
  const navigate = useNavigate();


  const [step2Billboard, setStep2Billboard] = useState("");
  const [step2SegmentedBillboard, setStep2SegmentedBillboard] = useState("");
  const [billboardId, setBillboardId] = useState("");
  const image = sessionStorage.getItem("downloadedImage");
  const billboardIdFromSession = sessionStorage.getItem("billboardId");
  const billboardImageFromSession = sessionStorage.getItem("billboardImage");


  // Retrieve the image from sessionStorage
  useEffect(() => {
    if (image && billboardIdFromSession && billboardImageFromSession) {
      setBillboardId(billboardIdFromSession)
      setStep2Billboard(billboardImageFromSession)
      setBanner(image);
      setCurrentStep(3);
    }
    return () => {
      sessionStorage.removeItem("billboardId");
      sessionStorage.removeItem("billboardImage");
      sessionStorage.removeItem("downloadedImage");
    };
  }, [image]);



  const fetchData = async () => {
    try {
      setLoading(true);

      if (currentStep === 1) {
        if (!billboard) {
          showNotification("Please upload a billboard image before proceeding.");
          return;
        }
        const billboardURL = await uploadImageToCloudinary(billboard);

        const response = await axios.post(
          `${apibaseurl}user/process-billboard`,
          {
            billboardImage: billboardURL,
            title,
            description
          },
          {
            withCredentials: true,
          }
        );

        if (response.status === 201) {
          setError(false);
          if (response.data.billboardImage && response.data.segmentedImage) {
            setBillboardId(response.data.id);
            setStep2Billboard(response.data.billboardImage);
            setStep2SegmentedBillboard(response.data.segmentedImage);
            sessionStorage.setItem("billboardId", response.data.id);
            sessionStorage.setItem("billboardImage", response.data.billboardImage);
          }
          setCurrentStep(2);
          setLoading(false);
        }
      }
      else if (currentStep === 3) {
        if (!banner) {
          showNotification("Please upload a banner image before proceeding.");
          return;
        }
        const bannerURL = await uploadImageToCloudinary(banner);
        console.log(bannerURL);
        console.log(billboardId);
        console.log(step2Billboard)
        const response = await axios.post(
          `${apibaseurl}user/process-banner`,
          {
            campaignId: billboardId,
            billboardImage: step2Billboard,
            bannerImage: bannerURL
          },
          {
            withCredentials: true,
          }
        );

        if (response.status === 201) {
          setError(false);
          // Store the image in sessionStorage
          if (response.data) {
            if (image) {
              sessionStorage.removeItem("downloadedImage");
              sessionStorage.removeItem("billboardId");
              sessionStorage.removeItem("billboardImage");
            }
            navigate('/success', { state: { finalBillboard: response.data.processedImage, billboardId } });
          }
        }
      }

    }
    catch (err) {
      console.error("API Error:", err);
      setError(true);
    } finally {
      setLoading(false);
    }
  };


  const handleStepOne = () => {
    if (!billboard) {
      showNotification("Please upload a billboard image before proceeding.");
    } else {
      fetchData();
      // setLoading(true)
      // setTimeout(() => {
      //   setCurrentStep(2);
      //   setLoading(false)
      // }, 6500);
    }
  };


  const handleStepTwo = () => {
    if (!step2Billboard) {
      setCurrentStep(1)
    } else {
      setCurrentStep(3);
    }
  };


  const handleStepThree = async () => {
    if (!banner) {
      showNotification("Please upload a banner image to proceed.");
      return;
    } else {
      fetchData();
      // setLoading(true)
      // setTimeout(() => {
      //   setCurrentStep(2);
      //   setLoading(false)
      //   navigate('/success')
      // }, 6500);
    }
  };



  return (
    <>
      <div className="bg-white min-h-screen flex ">
        {/* Sidebar */}
        <Sidebar />

        {/* Main Content */}
        <main className="flex-1 py-3 px-6">
          {/* Header */}
          <header className="flex justify-end items-center mb-6 w-full">

            {/* Create Task Button */}

          </header>

          {/* <div className="flex mt-2 mb-6">
            <h1 className="text-3xl">Upload Image</h1>
          </div> */}
          {error && (
            <Error errorMsg="Sorry, Some Internal server error has occured, Please try later" />
          )}
          {!error && (
            <div className=" mt-10 flex flex-col items-center justify-center">

              {/* Step 1 - Uploading Billboard Image */}
              {currentStep === 1 && (
                <div className="bg-white/50 border-2 border-gray-200 backdrop-blur-md rounded-lg p-8 max-w-lg text-center shadow-lg">
                  <h1 className="text-3xl text-gray-800 font-bold mb-4">  Upload Billboard Image   </h1>
                  <p className="text-lg text-gray-600 mb-6">
                    Upload the image that you want to use for the billboard.Once uploaded, proceed to the next step to upload yourbanner.
                  </p>
                  <FileUploader label="Choose Billboard Image" onFileChange={(file) => { setBillboard(file) }} />
                  <div
                    className="mt-3"
                  >
                    <input
                      type="text"
                      id="title"
                      placeholder="Title"
                      className="w-full p-2 mb-6 text-base text-gray-900 bg-[#F7F2FA] rounded-lg  border-b border-[#6750A4] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                    />
                    <input
                      type="description"
                      id="description"
                      placeholder="Description"
                      className="w-full p-2 mb-6 text-base text-gray-900 bg-[#F7F2FA] rounded-lg  border-b border-[#6750A4] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                    />
                  </div>
                  <Button label={loading ? "Processing..." : "Process with AI"} onClick={handleStepOne}
                    className={`mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg shadow ${loading
                      ? "opacity-50 cursor-not-allowed"
                      : "hover:bg-blue-700"
                      } focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    disabled={loading} />
                </div>
              )}

              {/* Step 2 - Show the Uploaded image and segemented image */}
              {currentStep === 2 && (
                <div className="max-w-[1450px] flex flex-col items-center gap-9">
                  <figure className="w-full flex justify-between gap-16">
                    <div className="text-center">
                      <img src={step2Billboard} alt="Source Image" className="h-[300px] rounded-lg" />
                      <figcaption className="mt-2 text-lg font-medium">Source Image</figcaption>
                    </div>
                    <div className="text-center">
                      <img src={step2SegmentedBillboard} alt="Segmented Image" className="h-[300px] rounded-lg" />
                      <figcaption className="mt-2 text-lg font-medium">Segemented Image</figcaption>
                    </div>
                  </figure>
                  <Button
                    label={loading ? "Just a moment, Processing..." : "Proceed To Next"}
                    onClick={handleStepTwo}
                    className={`mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg shadow ${loading
                      ? "opacity-50 cursor-not-allowed"
                      : "hover:bg-blue-700"
                      } focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    disabled={loading}
                  />
                </div>
              )}

              {/* Step 3 - Uploading banner Image */}
              {currentStep === 3 && (
                <div className="bg-white/50 border-2 border-gray-200 backdrop-blur-md rounded-lg p-8 max-w-lg text-center shadow-lg">
                  <h1 className="text-3xl text-gray-800 font-bold mb-4">
                    Upload Banner Image
                  </h1>
                  <p className="text-lg text-gray-600 mb-6">
                    Select the banner image you want to place on the billboard.
                    Once uploaded, you can proceed to process it with AI.
                  </p>
                  <p className="text-gray-800 text-center text-xl">Choose Banner Image </p>
                  <div className="w-full flex justify-center items-end gap-3">
                    <FileUploader
                      label=""
                      onFileChange={(file) => setBanner(file)}
                      bannerUploaded={banner}
                    />
                    <a href={`/create?billboardId=${billboardId}`}>
                      <button
                        className={`mt-4 px-1 min-w-[150px] h-14 py-3 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500`}
                      // onClick={() => navigate('/create')}
                      >
                        Create Banner
                      </button>
                    </a>
                  </div>
                  <Button
                    label={loading ? "Just a moment, Processing..." : "Process with AI"}
                    onClick={handleStepThree}
                    className={`mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg shadow ${loading
                      ? "opacity-50 cursor-not-allowed"
                      : "hover:bg-blue-700"
                      } focus:outline-none focus:ring-2 focus:ring-blue-500`}
                    disabled={loading}
                  />

                </div>
              )}
            </div>
          )}
        </main>
        {/* <img src={banner} alt="" /> */}
      </div>
    </>
  );
};

export default UploadImage;
