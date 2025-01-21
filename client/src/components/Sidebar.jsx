import React, { Component } from 'react'

export class Sidebar extends Component {
  render() {
    return (
      <aside className="w-50 fixed">
        <div className="px-2 flex items-center justify-center">
          <div className="w-24 h-24 text-white flex items-center justify-center font-bold">
            <img
              className="w-24 h-24"
              src='/logo2.jpg'
              alt="LOGO"
            />
          </div>
        </div>
        <nav className="">
          <ul className="space-y-2">
            <li className="px-6">
              <a
                href="/dashboard"
                className="w-fit flex items-center p-2 hover:bg-gray-100 rounded-lg"
                title="Dashboard"
              >
                <img className="w-9 h-9" src='/dashboard.svg' alt="Dashboard" />
                {/* <span className="ml-3 text-gray-700 font-semibold">Tasks</span> */}
              </a>
            </li>
            <li className="px-6">
              <a
                href="#"
                className="w-fit flex items-center p-2 hover:bg-gray-100 rounded-lg"
                title="Settings"
              >
                <img className="w-9 h-9" src='/settings.svg' alt="Settings" />
                {/* <span className="ml-3 text-gray-700 font-semibold">Settings</span> */}
              </a>
            </li>
            <li className="px-6">
              <a
                href="#"
                className="w-fit flex items-center p-2 hover:bg-gray-100 rounded-lg"
                title="Profile"
              >
                <img className="w-9 h-9" src='/account.svg' alt="Profile" />
                {/* <span className="ml-3 text-gray-700 font-semibold">Settings</span> */}
              </a>
            </li>
            <li className="px-6">
              <a
                href="/post"
                className="w-fit flex items-center p-2 hover:bg-gray-100 rounded-lg"
                title="Profile"
              >
                <img className="w-9 h-9" src='/post.png' alt="Post" />
                {/* <span className="ml-3 text-gray-700 font-semibold">Settings</span> */}
              </a>
            </li>
          </ul>
        </nav>
        <div className="mt-[35vh] px-6 py-4">
          <div className="">
            <a
              href="/"
              className="w-fit flex items-center p-2 hover:bggray-100 rounded-lg"
              title="Logout"
            >
              <img className="w-9 h-9" src='/logout.svg' alt="Logout" />
              {/* <span className="ml-3 text-gray-700 font-semibold">Settings</span> */}
            </a>
          </div>
        </div>
      </aside>
    )
  }
}

export default Sidebar