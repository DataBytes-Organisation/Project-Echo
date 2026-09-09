(function ($) {
  "use strict";

  function loadComponent(selector, url) {
    return new Promise((resolve, reject) => {
      $(selector).load(url, function (_response, status, xhr) {
        if (status === "error") {
          reject(new Error(`Failed to load ${url}: ${xhr.status}`));
          return;
        }

        resolve();
      });
    });
  }

  $(async function () {
    try {
      await Promise.all([
        loadComponent("#sidebar", "/admin/component/sidebar-component.html"),
        loadComponent("#header", "/admin/component/header-component.html"),
        loadComponent("#footer", "/admin/component/footer-component.html")
      ]);

      await $.getScript("/admin/js/app.min.js");
      await $.getScript("/admin/js/sidebarmenu.js");
    } catch (error) {
      console.error("Failed to initialise the admin layout:", error);
    }
  });
})(jQuery);