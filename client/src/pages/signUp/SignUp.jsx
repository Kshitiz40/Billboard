import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import apibaseurl from '../../apiConfig/api';

const SignUp = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [mobileNumber, setMobileNumber] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSignUp = async (e) => {
    e.preventDefault();
    // Add your signup logic here (e.g., API call)
    if (name && email && mobileNumber && password) {
      try{
        const response = await axios.post(
          `${apibaseurl}user/register`,
          {name,email,mobile_number : mobileNumber,password},
          {
            withCredentials : true,
          }
        );
  
        if(response.status === 201){
          navigate("/dashboard"); // Redirect to Home page after successful signup
        }
        else{
          alert("Something went wrong!");
        }
      }
      catch(error){
        alert(error.message);
      }

    } else {
      alert("Please fill in all fields.");
    }
  };

  const handleLogInRedirect = () => {
    navigate("/login");
  };

  return (
    <div
      className="flex items-center justify-center min-h-screen bg-cover bg-center"
      style={{ backgroundImage: "url('/Login.jpeg')" }} // Replace with your image path
    >
      <div className="bg-white w-96 py-12 px-12 shadow-lg rounded-2xl">
        {/* Header */}
        <h1 className="text-4xl font-normal text-gray-900 mb-2">Sign Up</h1>
        <p className="text-gray-600 mt-2 text-base mb-8">
          Create your account to get started
        </p>

        {/* Form */}
        <form onSubmit={handleSignUp} className="flex flex-col">
          {/* Name Input */}
          <label
            htmlFor="name"
            className="mb-1 text-base font-medium text-gray-700"
          >
            Enter your name
          </label>
          <input
            type="text"
            id="name"
            placeholder="Name"
            className="p-2 mb-4 text-base text-gray-900 bg-[#F7F2FA] rounded-lg border-b border-[#6750A4] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />

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
            className="p-2 mb-4 text-base text-gray-900 bg-[#F7F2FA] rounded-lg border-b border-[#6750A4] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          {/* Mobile Input */}
          <label
            htmlFor="mobile"
            className="mb-1 text-base font-medium text-gray-700"
          >
            Enter your Mobile
          </label>
          <input
            type="text"
            id="mobile"
            placeholder="Mobile"
            className="p-2 mb-4 text-base text-gray-900 bg-[#F7F2FA] rounded-lg border-b border-[#6750A4] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={mobileNumber}
            onChange={(e) => setMobileNumber(e.target.value)}
          />

          {/* Password Input */}
          <label
            htmlFor="password"
            className="mb-1 text-base font-medium text-gray-700"
          >
            Create a password
          </label>
          <input
            type="password"
            id="password"
            placeholder="Password"
            className="p-2 mb-6 text-base text-gray-900 bg-[#F7F2FA] rounded-lg border-b border-[#6750A4] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          {/* Sign Up Button */}
          <button
            type="submit"
            className="w-full bg-[#420d8b] text-white py-3 rounded-full text-sm font-medium shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2"
          >
            Sign Up
          </button>
        </form>

        {/* Log In Redirect */}
        <p className="mt-8 font-semibold text-sm text-slate-500">
          Already have an account?{" "}
          <button
            type="button"
            className="text-blue-500  hover:underline"
            onClick={handleLogInRedirect}
          >
            Log In
          </button>
        </p>
      </div>
    </div>
  );
};

export default SignUp;
