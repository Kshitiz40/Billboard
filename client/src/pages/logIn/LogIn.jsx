import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import showNotification from '../../components/PopupNotification';
import axios from "axios";
import apibaseurl from '../../apiConfig/api';

const LogIn = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    // Add your login logic here (e.g., API call)
    if (email && password) {
      try{
        const response = await axios.post(
          `${apibaseurl}user/login`,
          {email,password},
          {
            withCredentials : true,
          }
        );
  
        if(response.status === 200){
          navigate("/dashboard"); // Redirect to Home page after successful signup
        }
        else{
          alert("Something went wrong!");
        }
      }
      catch(error){
        alert("Invalid Credentials");
      }
    } else {
      alert("Please fill in all fields.");
    }
  };

  const handleSignUpRedirect = () => {
    navigate("/signup");
  };

  return (
    <div
      className="flex  items-center justify-center min-h-screen bg-cover bg-center"
      style={{ backgroundImage: "url('/Login.jpeg')" }} // Corrected image path
    >
      <div className="bg-white w-96 py-12 px-12 shadow-lg   rounded-2xl  ">
        {/* Header */}
        <h1 className="text-4xl font-normal text-gray-900 mb-2">Log In</h1>
        <p className="text-gray-600 mt-2 text-base  mb-8">
          Enter your details to continue
        </p>

        {/* Form */}
        <form onSubmit={handleLogin} className="flex flex-col ">
          {/* Email Input */}
          <label
            htmlFor="email"
            className="mb-1 text-base font-medium text-gray-700"
          >
            Enter your email
          </label>
          <input
            type="email"
            id="email"
            placeholder="Email"
            className="p-2 mb-6 text-base text-gray-900 bg-[#F7F2FA] rounded-lg  border-b border-[#6750A4] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          {/* Password Input */}
          <label
            htmlFor="password"
            className="mb-1 text-base font-medium text-gray-700"
          >
            Enter your password
          </label>
          <input
            type="password"
            id="password"
            placeholder="Password"
            className="p-2 mb-6 text-base text-gray-900 bg-[#F7F2FA] rounded-lg border-b border-[#6750A4] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          {/* Remember Me & Forgot Password */}
          <div className="flex  items-center justify-between mb-6">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="rememberMe"
                className="w-5 h-5 bg-[#420d8b] text-blue-600 border-[#420d8b] rounded"
              />
              <label
                htmlFor="rememberMe"
                className="ml-2 text-sm text-gray-700"
              >
                Remember Me
              </label>
            </div>
            <a href="#" className="text-sm text-blue-500 hover:underline">
              Forgot Password?
            </a>
          </div>

          {/* Login Button */}
          <button
            type="submit"
            className="w-full bg-[#420d8b] text-white py-2 rounded-full text-sm font-medium shadow-md focus:outline-none focus:ring-2  focus:ring-offset-2"
          >
            Log In
          </button>
        </form>

        {/* Sign Up Redirect */}
        <p className="mt-6 font-semibold text-sm  text-slate-500">
          Don't have an account?{" "}
          <button
            type="button"
            className="text-blue-500 ml-1 hover:underline"
            onClick={handleSignUpRedirect}
          >
            Sign Up
          </button>
        </p>
      </div>
    </div>
  );
};

export default LogIn;
