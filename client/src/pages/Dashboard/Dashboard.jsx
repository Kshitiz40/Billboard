import React, { Profiler, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "../../components/Sidebar";
import apibaseurl from '../../apiConfig/api';
import axios from "axios";

const Dashboard = () => {
  const navigate = useNavigate();
  const [userData, setUserData] = useState(null);
  const [initialLoader, setInitialLoader] = useState(false);

  const getUser = async () => {
    try {
      const response = await axios.get(
        `${apibaseurl}user/`,
        {
          withCredentials: true,
        }
      );
      if (response.status === 200) {
        setInitialLoader(true);
      }
    }
    catch (error) {
      navigate('/Login');
    }
  }

  const getBillboardInfo = async () => {
    try {
      const response = await axios.get(
        `${apibaseurl}user/getCampaigns`,
        {
          withCredentials: true,
        }
      );
      if (response.status === 200) {
        setUserData(response.data.campaigns);
      }
    }
    catch (error) {
      console.log("no proccessed image found")
    }
  }

  useEffect(() => {
    getUser()
    getBillboardInfo();
  }, []);

  return (
    initialLoader && <div className="bg-white min-h-screen flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="mr-10 ml-28 flex-1 p-6">
        {/* Header */}
        <header className="flex justify-between items-center mb-6">
          <div className="flex items-center rounded-full w-full max-w-2xl text-3xl">
            <h1>Dashboard</h1>
          </div>

          {/* Create Task Button */}
          <button
            onClick={() => {
              navigate("/upload")
            }}
            className="bg-purple-600 text-white px-6 py-2 rounded-full font-semibold shadow-lg hover:bg-purple-800"
          >
            Create Task
          </button>
        </header>

        {/* Dashboard Content */}
        <div className="grid grid-cols-3 gap-16">

          {/* TOOLS */}
          <div className="bg-white rounded-lg shadow-lg  p-2 h-fit">
            <div className="p-4">
              {/* Create Banner Section */}
              <div className="bg-white rounded-lg shadow-lg p-4 items-start mb-4">
                <h2 className="text-xl font-semibold mb-4">Create Your Custom Banner</h2>
                <p className="text-gray-600 mb-6">
                  Design and personalize your banners from scratch with an intuitive editor. Add text, images, and colors to make your banner truly unique.
                </p>
                <button
                  onClick={() => navigate("/draw")}
                  className="bg-purple-600 text-white px-6 py-2 rounded-full font-medium shadow-md hover:bg-purple-700"
                >
                  Start Designing
                </button>
              </div>

              {/* Upload Images Section */}
              <div className="bg-white rounded-lg shadow-lg p-4 items-start">
                <h2 className="text-xl font-semibold mb-4">AI-Powered Billboard Transformation</h2>
                <p className="text-gray-600 mb-6">
                  Seamlessly replace the banner in your billboard image using AI. Upload your billboard and banner images, and let AI handle the precise alignment and blending.
                </p>
                <button
                  onClick={() => { navigate("/upload") }}
                  className="bg-purple-600 text-white px-6 py-2 rounded-full font-medium shadow-md hover:bg-purple-700"
                >
                  Transform with AI
                </button>
              </div>
            </div>
          </div>


          {/* Processed Images */}
          <div className="bg-white rounded-lg shadow-lg p-4 h-fit">
            <h2 className="text-2xl mb-4">Processed Images</h2>
            {userData && userData.length === 0 && <div className="bg-gray-100 p-5 rounded-2xl">
              No Images Processed yet!
            </div>
            }
            {userData && userData.length > 0 && <ul className="space-y-4 bg-gray-100 p-5 rounded-2xl">
              {userData.map((item, index) => (
                <li key={index} className="flex items-center space-x-4" onClick= {() => navigate('/campaign', {state : {billboardId : item.id}})}>
                  <div className="w-16 h-16 rounded-full bg-gray-300">
                    <img src={item.final_image} className="w-16 h-16 rounded-full" />
                  </div>
                  <div>
                    <h3 className="font-semibold">{item.title}</h3>
                    {item.description}
                  </div>
                </li>
              ))}
            </ul>}
          </div>

          {/* Statistics */}
          <div className="bg-white rounded-lg shadow-lg p-4 h-fit">
            <h2 className="text-2xl mb-4">Statistics</h2>
            <ul className="space-y-4 p-0">
              <li className="flex items-center bg-gray-100 rounded-2xl p-4 border-2 border-gray-200">
                <span className="font-semibold text-gray-700 mr-7 mb-3">
                  <img src="./credits.svg" alt="credits" />
                </span>
                <span className="">
                  <div className="text-4xl">100</div>
                  <div className="text-lg">Remaining Credits</div>
                </span>
              </li>
              <li className="flex items-center bg-gray-100 rounded-2xl p-4 border-2 border-gray-200">
                <span className="font-semibold text-gray-700 mr-7 mb-3">
                  <img src='./credits.svg' alt="pucture" />
                </span>
                <span className="">
                  <div className="text-4xl">{userData && userData.length}</div>
                  <div className="text-lg">Processed Images</div>
                </span>
              </li>
              <li className="flex items-center bg-gray-100 rounded-2xl p-4 border-2 border-gray-200">
                <span className="font-semibold text-gray-700 mr-7 mb-3">
                  <img src="./ChatGptCredits.svg" alt="credits" />
                </span>
                <span className="">
                  <div className="text-4xl">0</div>
                  <div className="text-lg">Banners Created</div>
                </span>
              </li>
              <li className="flex items-center bg-gray-100 rounded-2xl p-4 border-2 border-gray-200">
                <span className="font-semibold text-gray-700 mr-7 mb-3">
                  <img src="./ChatGptCredits.svg" alt="credits" />
                </span>
                <span className="">
                  <div className="text-4xl">{userData && userData.length}</div>
                  <div className="text-lg">Billboards Transformed</div>
                </span>
              </li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;